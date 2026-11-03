"""Read robot state, or perform an explicitly requested 2-degree wrist-only test."""
import argparse
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

p = argparse.ArgumentParser()
p.add_argument('--execute', action='store_true')
a = p.parse_args()
r = SO101Follower(SO101FollowerConfig(port='/dev/tty.usbmodem5B421342761', id='my_awesome_follower_arm', use_degrees=True))
log = {'execute': a.execute, 'samples': [], 'status': 'precheck'}
active = False
r.bus.connect()
try:
    if not r.calibration or not r.is_calibrated:
        raise RuntimeError('Calibration missing or mismatched')
    q = r.bus.sync_read('Present_Position')
    torque = r.bus.sync_read('Torque_Enable')
    mode = r.bus.read('Operating_Mode', 'wrist_roll')
    time.sleep(.4)
    second = r.bus.sync_read('Present_Position')
    log.update(initial=q, torque=torque, wrist_mode=mode, second=second)
    print(json.dumps(log, indent=2), flush=True)
    if mode != 0:
        raise RuntimeError('Wrist is not in position mode')
    if max(abs(q[n]-second[n]) for n in q) > .5:
        raise RuntimeError('Robot is not stationary')
    if a.execute:
        import math
        joint = ET.parse(Path(__file__).with_name('so101_new_calib.urdf')).getroot().find("joint[@name='wrist_roll']")
        lower = math.degrees(float(joint.find('limit').get('lower'))) + 3
        upper = math.degrees(float(joint.find('limit').get('upper'))) - 3
        start = float(second['wrist_roll'])
        delta = 2. if start + 2 <= upper else -2.
        if not lower <= start <= upper or not lower <= start+delta <= upper:
            raise RuntimeError('Wrist too close to joint limit')
        # Only wrist_roll receives commands; no controller reconfiguration.
        r.bus.write('Goal_Position', 'wrist_roll', start)
        active = True
        r.bus.enable_torque('wrist_roll')
        for source, target in [(start, start+delta), (start+delta, start)]:
            for i in range(1, 61):
                u = i/60
                goal = source+(target-source)*(3*u*u-2*u*u*u)
                r.bus.write('Goal_Position', 'wrist_roll', goal)
                measured = r.bus.sync_read('Present_Position')
                if abs(measured['wrist_roll']-goal) > 2:
                    raise RuntimeError('Wrist tracking error exceeds 2 degrees')
                if any(abs(measured[n]-second[n]) > 1 for n in second if n != 'wrist_roll'):
                    raise RuntimeError('Another joint moved; test stopped')
                log['samples'].append({'goal':goal, 'measured':measured['wrist_roll']})
                time.sleep(.05)
            time.sleep(.3)
            measured = r.bus.read('Present_Position', 'wrist_roll')
            print('Target:',target,'Measured:',measured,flush=True)
            if abs(measured-target) > .8:
                raise RuntimeError('Wrist did not reach target')
        if not torque['wrist_roll']:
            r.bus.disable_torque('wrist_roll')
        log['status']='completed'
    else:
        log['status']='read_only_complete'
except BaseException as e:
    log['status']='aborted'
    log['error']=str(e)
    if active:
        try:
            current=r.bus.read('Present_Position','wrist_roll')
            r.bus.write('Goal_Position','wrist_roll',current)
            log['abort_action']='hold current wrist position; torque left on'
        except Exception as hold_error:
            log['hold_error']=str(hold_error)
    raise
finally:
    Path('wrist_motion_result.json' if a.execute else 'motion_precheck.json').write_text(json.dumps(log,indent=2))
    r.bus.disconnect(disable_torque=False)
