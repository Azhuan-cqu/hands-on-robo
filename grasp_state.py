"""Per-robot process lock and atomic grasp journal, independent of --output."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time


class GraspState:
    def __init__(self, config, directory=None):
        identity = str(config['robot_id'])
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        root = Path(directory) if directory else Path.home()/'.local/state/hands-on-robo'
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / (key + '.json')
        self.lock = open(root / (key + '.lock'), 'a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise RuntimeError('Another process controls this robot; startup blocked')
        self.identity = identity

    def read(self):
        try:
            record = json.loads(self.path.read_text())
            if record['robot_id'] != self.identity or record['state'] not in ('empty','holding','unknown'):
                raise ValueError('Invalid state')
            if record['state'] == 'holding':
                for key in ('held_command', 'opening'):
                    if type(record[key]) not in (int, float) or not math.isfinite(record[key]):
                        raise ValueError('Invalid holding record')
            return record
        except FileNotFoundError:
            return {'state':'empty', 'reason':'default_initial_empty'}
        except (OSError, ValueError, KeyError, TypeError):
            return {'state':'unknown', 'reason':'unreadable state journal'}

    def write(self, state, **details):
        record = dict(details, robot_id=self.identity, state=state, time=time.time())
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w') as stream:
            json.dump(record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def close(self):
        self.lock.close()


def startup_decision(previous, confirmed, recover=False):
    """Default initial/recorded empty; never overwrite a holding or unknown record implicitly."""
    if confirmed is None and not recover and previous.get('state') == 'empty':
        return 'empty'
    if confirmed not in ('empty', 'holding'):
        raise ValueError('Startup state unknown: inspect the gripper; no robot actions allowed')
    if recover:
        if confirmed != 'holding' or previous.get('state') != 'holding':
            raise ValueError('Recovery needs confirmed holding and a saved holding record')
        if previous.get('held_command') is None or previous.get('opening') is None:
            raise ValueError('No saved grip command/opening; automatic recovery unavailable')
        return 'recover'
    if confirmed == 'holding':
        raise ValueError('Object held: normal pick/teach blocked. Use --recover-place or stop')
    return 'empty'
