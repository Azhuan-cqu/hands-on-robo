"""Offline diagnosis only: compare range midpoint with calibration reference pose.
The reference pose is not proven to be the URDF zero. Never auto-apply this mapping.
"""
import argparse
import copy
import json
from pathlib import Path
from handeye_calibration import URDFFK, solve
from camera_intrinsic import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motor-calibration', type=Path, required=True)
    p.add_argument('--samples', type=Path, default=Path('handeye_output/samples.json'))
    p.add_argument('--urdf', type=Path, default=Path('so101_new_calib.urdf'))
    p.add_argument('--output', type=Path, default=Path('joint_zero_comparison.json'))
    a = p.parse_args()
    cal = json.loads(a.motor_calibration.read_text())
    data = json.loads(a.samples.read_text())
    fk = URDFFK(a.urdf)
    # Verified against this installation's Feetech STS3215 driver:
    # max_res = 4096-1; homing reference = int(max_res/2) = 2047.
    resolution_minus_one = 4095
    reference = 2047
    offsets = {n: ((cal[n]['range_min']+cal[n]['range_max'])/2-reference)*360/resolution_minus_one for n in fk.names}
    report = {'applied_to_robot': False, 'applied_to_runtime_config': False,
              'assumption': 'Current motor calibration matches the file used during sampling; samples do not contain a motor calibration snapshot.',
              'candidate_note': 'Homing reference pose is only a hypothesis for URDF zero, not an identified zero alignment.',
              'mapping': 'q_candidate_deg = q_lerobot_deg + offset_deg; motor_command_deg = q_candidate_deg - offset_deg',
              'candidate_offsets_deg': offsets, 'results': {}}
    for name, shift in [('midpoint_current',dict.fromkeys(fk.names,0)),('homing_reference_candidate',offsets)]:
        rows = copy.deepcopy(data['samples'])
        for row in rows:
            row['T_base_gripper'] = fk({n:row['joints_deg'][n]+shift[n] for n in fk.names}).tolist()
        result = solve(rows)
        report['results'][name] = {k:result[k] for k in ['board_position_rms_mm','board_rotation_rms_deg','T_gripper_camera']}
    save_json(a.output,report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
