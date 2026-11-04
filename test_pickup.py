"""Pickup output is driven by measured lift and retained contact, never a sent command."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pick_place as pp
from grasp_state import GraspState
from test_retries import Bus, FakePlanner, stages, observation


class PickupEvidenceTests(unittest.TestCase):
    def assess(self, openings=None, heights=None, span=0, reference=None):
        return pp.assess_pickup([30.]*8 if openings is None else openings, 26., 0.,
            {'opening':30., 'tcp_z_m':.1} if reference is None else reference,
            [.12]*8 if heights is None else heights, span, {})

    def test_retained_contact_and_lift(self):
        for opening in (28.,30.,32.):
            self.assertTrue(self.assess(openings=[opening]*8)['verified'])
        self.assertTrue(self.assess(heights=[.11]*8)['verified'])

    def test_insufficient_lift_or_arm_movement(self):
        self.assertEqual(self.assess(heights=[.105]*8)['reason'],'insufficient_measured_lift')
        self.assertEqual(self.assess(span=.7)['reason'],'arm_not_stable_after_lift')

    def test_lost_grip_unstable_grip_or_large_change(self):
        for openings in ([0.]*8,[26.]*8,[30.,31.]*4):
            self.assertEqual(self.assess(openings=openings)['reason'],'contact_not_retained')
        self.assertEqual(self.assess(openings=[33.]*8)['reason'],'opening_changed_after_lift')

    def test_unknown_and_invalid_feedback(self):
        self.assertEqual(self.assess(openings=[30.]*4)['reason'],'insufficient_samples')
        self.assertEqual(self.assess(heights=[float('nan')]*8)['reason'],'non_finite_feedback')
        self.assertFalse(pp.assess_pickup([30.]*8,26,0,None,[.12]*8,0,{})['verified'])


class AfterLiftBus(Bus):
    def __init__(self, after_opening):
        super().__init__()
        self.after_opening=after_opening
    def sync_read(self, register):
        q=super().sync_read(register)
        if q['wrist_roll'] >= 10:
            q['gripper']=self.after_opening
        return q


class PickupIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)
        for name,value in [('imshow',None),('waitKey',-1),('imwrite',True)]:
            self.enterContext(patch.object(pp.cv2,name,return_value=value))
        self.enterContext(patch.object(pp.time,'sleep'))
        self.enterContext(patch.object(pp,'recovery_context',return_value='test'))
        self.cap=SimpleNamespace(read=lambda:(True,np.zeros((8,8,3),np.uint8)))
        self.store=GraspState({'robot_id':'pickup-test'},self.out/'journal')
        self.addCleanup(self.store.close)
        self.stdout=io.StringIO()

    def run_plan(self,bus,planner=None,recovery=None):
        planner=planner or FakePlanner()
        plan=stages(planner)
        if recovery:
            plan=[s for s in plan if s['stage'] in ('lift','transfer','release','retract')]
        elif planner.c.get('pick_only'):
            plan=[s for s in plan if s['stage'] not in ('transfer','release','retract')]
        robot=SimpleNamespace(bus=bus,configure=lambda:None)
        with redirect_stdout(self.stdout):
            pp.execute(robot,planner,plan,observation(),self.cap,self.out,
                       state_store=self.store,recovery=recovery)

    def log(self):
        return json.loads((self.out/'execution.json').read_text())

    def test_pick_only_prints_true_after_lift_and_persists_evidence(self):
        planner=FakePlanner(); planner.c['pick_only']=True
        self.run_plan(Bus(),planner)
        log=self.log()
        self.assertTrue(log['cube_picked_up'])
        self.assertEqual(log['status'],'lift_feedback_verified_holding')
        names=[e['stage'] for e in log['events']]
        self.assertLess(names.index('lift'),names.index('pickup_verified'))
        self.assertIn('cube_picked_up = True',self.stdout.getvalue())
        self.assertTrue(self.store.read()['cube_picked_up'])
        self.assertAlmostEqual(self.store.read()['grasp_reference']['tcp_z_m'],.1)

    def test_contact_without_lift_never_reports_true(self):
        with self.assertRaises(pp.MotionFailure):
            self.run_plan(Bus(fail_lift=True))
        self.assertFalse(self.log()['cube_picked_up'])
        self.assertNotIn('cube_picked_up = True',self.stdout.getvalue())

    def test_lost_object_after_lift_stops_before_transfer_and_holds(self):
        bus=AfterLiftBus(0.)
        with self.assertRaisesRegex(RuntimeError,'contact_not_retained'):
            self.run_plan(bus)
        log=self.log()
        self.assertFalse(log['cube_picked_up'])
        self.assertNotIn('transfer',[e['stage'] for e in log['events']])
        contact=next(e for e in log['events'] if e['stage']=='possible_contact')
        self.assertEqual(bus.commands[-1]['gripper'],contact['held_command'])
        self.assertNotIn('cube_picked_up = True',self.stdout.getvalue())

    def test_changed_opening_after_lift_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError,'opening_changed_after_lift'):
            self.run_plan(AfterLiftBus(33.))
        self.assertFalse(self.log()['cube_picked_up'])

    def test_too_little_actual_lift_never_transfers(self):
        planner=FakePlanner(); tcp=planner.tcp
        def reduced_height(q):
            T=tcp(q); T[2,3]=.1+q[0]/2000
            return T
        planner.tcp=reduced_height
        with self.assertRaisesRegex(RuntimeError,'insufficient_measured_lift'):
            self.run_plan(Bus(),planner)
        self.assertFalse(self.log()['cube_picked_up'])
        self.assertNotIn('transfer',[e['stage'] for e in self.log()['events']])

    def test_pickup_is_a_milestone_not_placement_success(self):
        self.run_plan(Bus())
        self.assertTrue(self.log()['cube_picked_up'])
        self.assertEqual(self.log()['status'],'placement_not_verified')

    def test_legacy_recovery_does_not_invent_pickup_evidence(self):
        self.run_plan(Bus(),recovery={'held_command':26,'opening':30})
        self.assertFalse(self.log()['cube_picked_up'])
        self.assertEqual(self.log()['pickup_verification']['reason'],'missing_grasp_reference')
        self.assertNotIn('cube_picked_up = True',self.stdout.getvalue())

    def test_recovery_rechecks_saved_grasp_reference(self):
        self.run_plan(Bus(),recovery={'held_command':26,'opening':30,
                      'grasp_reference':{'opening':30,'tcp_z_m':.08}})
        self.assertTrue(self.log()['cube_picked_up'])
        self.assertAlmostEqual(self.log()['pickup_verification']['measured_lift_mm'],30.)


if __name__=='__main__':
    unittest.main()
