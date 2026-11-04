"""Offline failure injection: no camera, serial port, or robot required."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pick_place as pp


class Bus:
    def __init__(self, empty=0, fail_lift=False):
        self.q = {'wrist_roll': 0., 'gripper': 90.}
        self.commands = []
        self.empty = empty
        self.fail_lift = fail_lift
    def sync_read(self, register):
        return self.q.copy()
    def sync_write(self, register, command):
        self.commands.append(command.copy())
        if not (self.fail_lift and command.get('wrist_roll', 0) > 1):
            self.q['wrist_roll'] = command.get('wrist_roll', self.q['wrist_roll'])
        if 'gripper' in command:
            self.q['gripper'] = max(command['gripper'], 30) if not self.empty else command['gripper']
    def write(self, register, name, value):
        self.commands.append({name: value})
        self.q[name] = value if self.empty else max(value, 30.)
        if self.empty and value <= 2:
            self.empty -= 1
    def enable_torque(self, **kw):
        pass


class FakePlanner:
    def __init__(self):
        self.c = dict(auto=True, pick_only=False, speed_deg_s=15, gripper_open=90,
                      gripper_closed=0, table_z_m=0, clearance_m=.05,
                      cube={'side_mm':25}, retry={'grasp_attempts':3})
        self.fk = SimpleNamespace(names=['wrist_roll'])
        self.lo, self.hi = np.array([-90]), np.array([90])
        self.replans = 0
    def clearance(self, q):
        pass
    def tcp(self, q):
        T = np.eye(4); T[2, 3] = .1 + q[0]/1000
        return T
    def segment(self, start, goal, seed):
        q = np.array([(goal[2,3]-.1)*1000])
        return [q.tolist()], q
    def refine(self, plan, delta):
        return copy.deepcopy(plan)
    def plan(self, obs):
        self.replans += 1
        return stages(self)


def stages(planner):
    return [dict(stage=name, joints_deg=[[q]], T_base_tcp=planner.tcp([q]).tolist(),
                 gripper=90 if name in ('approach','hover','descend','release','retract') else 0)
            for name,q in [('approach',0),('hover',0),('descend',0),('grasp',0),('lift',10),('transfer',10),('release',10),('retract',10)]]


def observation():
    return dict(joints_deg={'wrist_roll':0.}, gripper=90,
                targets={'cube':{'T_base_marker':np.eye(4).tolist()},
                         'plate':{'T_base_marker':np.eye(4).tolist()}})


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        for name, value in [('imshow',None),('waitKey',-1),('imwrite',True)]:
            patcher = patch.object(pp.cv2, name, return_value=value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(pp.time, 'sleep'); patcher.start(); self.addCleanup(patcher.stop)
        self.cap = SimpleNamespace(read=lambda:(True,np.zeros((8,8,3),np.uint8)))
    def run_execution(self, bus, planner, observation_effect=None):
        robot = SimpleNamespace(bus=bus, configure=lambda:None)
        scene = SimpleNamespace(c=planner.c, fk=planner.fk)
        with patch.object(pp,'observe', side_effect=observation_effect, return_value=observation()):
            pp.execute(robot,planner,stages(planner),observation(),self.cap,self.out,scene)
    def test_empty_grasp_reacquires_then_succeeds(self):
        bus, planner = Bus(empty=1), FakePlanner()
        self.run_execution(bus,planner)
        log=json.loads((self.out/'execution.json').read_text())
        self.assertEqual(planner.replans,1)
        names=[e['stage'] for e in log['events']]
        self.assertLess(names.index('empty_grasp_retry'),names.index('retry_retreat'))
        self.assertLess(names.index('retry_retreat'),names.index('lift'))
        self.assertEqual(log['status'],'placement_not_verified')
    def test_empty_grasp_exhaustion_never_transfers(self):
        bus, planner = Bus(empty=10), FakePlanner()
        with self.assertRaisesRegex(RuntimeError,'Retry limit'):
            self.run_execution(bus,planner)
        log=json.loads((self.out/'execution.json').read_text())
        self.assertEqual(sum(e['stage']=='empty_grasp_retry' for e in log['events']),3)
        self.assertNotIn('transfer',[e['stage'] for e in log['events']])
        self.assertEqual(planner.replans,2)
    def test_loaded_motion_failure_keeps_preload_no_release(self):
        bus, planner = Bus(fail_lift=True), FakePlanner()
        with self.assertRaises(pp.MotionFailure):
            self.run_execution(bus,planner)
        log=json.loads((self.out/'execution.json').read_text())
        contact=next(e for e in log['events'] if e['stage']=='possible_contact')
        self.assertEqual(bus.commands[-1]['gripper'],contact['held_command'])
        self.assertTrue(log['holding_possible_object'])
        self.assertNotIn('release',[e['stage'] for e in log['events']])
        self.assertEqual(planner.replans,0)
    def test_pregrasp_unstable_retreat_reobserve(self):
        bus, planner = Bus(), FakePlanner()
        calls=[]
        def obs(*a,**kw):
            calls.append(a[3].name)
            if calls.count('pregrasp')==1 and calls[-1]=='pregrasp':
                raise pp.ObservationRejected('unstable')
            return observation()
        self.run_execution(bus,planner,obs)
        self.assertEqual(planner.replans,1)
        self.assertIn('retry_cube',calls)
    def test_pregrasp_exhaustion_never_descends(self):
        bus, planner = Bus(), FakePlanner()
        def obs(*a,**kw):
            if a[3].name=='pregrasp':
                raise pp.ObservationRejected('unstable')
            return observation()
        with self.assertRaisesRegex(RuntimeError,'Retry limit'):
            self.run_execution(bus,planner,obs)
        log=json.loads((self.out/'execution.json').read_text())
        self.assertNotIn('descend',[e['stage'] for e in log['events']])
    def test_observation_missing_is_bounded_and_scans_twice(self):
        scene=SimpleNamespace(c={'auto':True,'pick_only':True},fk=SimpleNamespace(names=['wrist_roll']),detect=lambda f,q:({},f))
        robot=SimpleNamespace(bus=Bus())
        with patch.object(pp,'wrist_search') as scan:
            with self.assertRaises(pp.ObservationRejected):
                pp.observe(self.cap,robot,scene,self.out,scan=scan)
            self.assertEqual(scan.call_count,2)
    def test_observation_success_after_scan(self):
        scene=SimpleNamespace(c={'auto':True})
        with patch.object(pp,'stable_observation',side_effect=[pp.ObservationRejected('missing'),(observation(),np.zeros((8,8,3),np.uint8))]), patch.object(pp,'wrist_search') as scan:
            result=pp.observe(self.cap,None,scene,self.out,scan=scan)
            self.assertIn('cube',result['targets'])
            scan.assert_called_once_with(0)
    def test_unstable_pose_batch_rejected(self):
        counter=iter([0,.02,0,.02,0])
        def detect(f,q):
            T=np.eye(4); T[0,3]=next(counter)
            return {'cube':{'T_base_marker':T.tolist()}},f
        scene=SimpleNamespace(c={'pick_only':True},fk=SimpleNamespace(names=['wrist_roll']),detect=detect)
        with self.assertRaisesRegex(pp.ObservationRejected,'position unstable'):
            pp.stable_observation(self.cap,SimpleNamespace(bus=Bus()),scene)
    def test_search_preflights_before_writing(self):
        planner=FakePlanner(); bus=Bus()
        planner.clearance=lambda q: (_ for _ in ()).throw(ValueError('collision')) if q[0]>5 else None
        with self.assertRaisesRegex(ValueError,'collision'):
            pp.wrist_search(SimpleNamespace(bus=bus),planner,{'wrist_roll':0},0,lambda:None)
        self.assertEqual(bus.commands,[])
    def test_search_absolute_offsets_and_cancel(self):
        planner=FakePlanner(); bus=Bus()
        for attempt,target in [(0,12),(1,-12)]:
            pp.wrist_search(SimpleNamespace(bus=bus),planner,{'wrist_roll':0},attempt,lambda:None)
            self.assertEqual(bus.q['wrist_roll'],target)
        with self.assertRaises(KeyboardInterrupt):
            pp.wrist_search(SimpleNamespace(bus=bus),planner,{'wrist_roll':0},0,lambda:(_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(bus.q['wrist_roll'],-12)
    def test_refine_preserves_tray_destination(self):
        planner=FakePlanner(); plan=stages(planner)
        changed=pp.Planner.refine(planner,plan,np.array([.01,0,0]))
        for old,new in zip(plan,changed):
            if old['stage'] in ('transfer','release','retract'):
                self.assertEqual(old['T_base_tcp'],new['T_base_tcp'])
        self.assertAlmostEqual(changed[1]['T_base_tcp'][0][3],.01)

if __name__=='__main__':
    unittest.main()
