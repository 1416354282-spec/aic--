#!/usr/bin/env python3
"""Watch a recorded Linux training process; pause its group before disks fill."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

GIB = 1024 ** 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_identity(pid, expected_config, proc_root=Path('/proc')):
    directory = Path(proc_root) / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        arguments = (directory / 'cmdline').read_bytes().decode().split('\0')
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] in ('Z', 'X'):
        return None
    if not any(arg.endswith('/train.py') for arg in arguments):
        raise RuntimeError('Recorded PID no longer runs train.py; refusing to signal it')
    if '--config' not in arguments or arguments[arguments.index('--config') + 1] != str(expected_config):
        raise RuntimeError('Training configuration does not match the recorded process')
    return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0]}


def disk_decision(disks, paused_by_guard):
    if paused_by_guard:
        return 'resume' if all(d['free_bytes'] >= d['resume_bytes'] for d in disks) else 'keep'
    return 'pause' if any(d['free_bytes'] < d['pause_bytes'] for d in disks) else 'keep'


def signal_training(identity, config, signum):
    actual = process_identity(identity['pid'], config)
    if actual is None or actual['start_ticks'] != identity['start_ticks']:
        raise RuntimeError('Process exited or PID was reused; refusing to signal it')
    if os.getpgid(identity['pid']) != identity['pid']:
        raise RuntimeError('Training does not own its process group; refusing to signal it')
    os.killpg(identity['pid'], signum)


def watch(record, status, interval=30):
    record, status = Path(record), Path(status)
    previous = json.loads(status.read_text()) if status.exists() else {}
    owner = previous.get('paused_process')
    rules = [('/', 6, 8), ('/root/autodl-fs', 12, 16), ('/root/autodl-tmp', 6, 8)]
    last_event = previous.get('last_event')
    while True:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            launch = json.loads(record.read_text())['launch']
            command = launch['command']
            config = command[command.index('--config') + 1]
            identity = process_identity(int(launch['pid']), config)
            if identity is None:
                write_json(status, {'guard_pid': os.getpid(), 'checked_at': timestamp,
                                    'state': 'training_exited', 'training_pid': launch['pid'],
                                    'last_event': last_event})
                return
            if owner and (owner['pid'], owner['start_ticks']) != (identity['pid'], identity['start_ticks']):
                raise RuntimeError('Recorded run changed while an older run is paused; manual review needed')
            disks = []
            for path, floor, resume in rules:
                usage = shutil.disk_usage(path)
                disks.append({'path': path, 'free_bytes': usage.free, 'total_bytes': usage.total,
                              'pause_bytes': floor * GIB, 'resume_bytes': resume * GIB})
            decision = disk_decision(disks, owner is not None)
            if decision == 'pause':
                # Only resume processes paused by this guard, never a user's manual pause.
                if identity['state'] not in ('T', 't'):
                    signal_training(identity, config, signal.SIGSTOP)
                    owner = identity
                    last_event = {'action': 'paused_low_disk', 'at': timestamp}
            elif decision == 'resume':
                signal_training(identity, config, signal.SIGCONT)
                owner = None
                last_event = {'action': 'resumed_disk_recovered', 'at': timestamp}
            write_json(status, {'guard_pid': os.getpid(), 'checked_at': timestamp,
                                'state': 'paused_low_disk' if owner else 'monitoring',
                                'training_pid': launch['pid'], 'paused_process': owner,
                                'run_dir': launch['run_dir'], 'disks': disks, 'last_event': last_event})
        except Exception as error:
            write_json(status, {'guard_pid': os.getpid(), 'checked_at': timestamp,
                                'state': 'monitor_error', 'error': str(error),
                                'paused_process': owner, 'last_event': last_event})
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')
    watch(args.record, args.status, args.interval)
