import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pick_place as pp
from grasp_state import GraspState, startup_decision
from test_retries import Bus, FakePlanner, stages, observation


class StartupStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = GraspState({'robot_id':'test'}, self.root/'state')
        self.addCleanup(self.store.close)

    def test_missing_defaults_empty_but_corruption_blocks(self):
        self.assertEqual(self.store.read()['state'], 'empty')
        self.assertEqual(startup_decision(self.store.read(), None), 'empty')
        self.store.path.write_text('broken json')
        self.assertEqual(self.store.read()['state'], 'unknown')
        for confirmed in (None, 'unknown', 'holding'):
            with self.assertRaises(ValueError):
                startup_decision(self.store.read(), confirmed)

    def test_lock_and_persistence_across_restart(self):
        self.store.write('holding', held_command=26, opening=30)
        with self.assertRaisesRegex(RuntimeError, 'Another process'):
            GraspState({'robot_id':'test'}, self.root/'state')
        self.store.close()
        new = GraspState({'robot_id':'test'}, self.root/'state')
        self.addCleanup(new.close)
        self.assertEqual(new.read()['held_command'], 26)
        self.assertEqual(startup_decision(new.read(), 'holding', True), 'recover')
        with self.assertRaises(ValueError):
            startup_decision(new.read(), 'holding', False)
        self.assertEqual(startup_decision(new.read(), 'empty'), 'empty')

    def test_recovery_requires_record_and_fresh_confirmation(self):
        for record in ({'state':'unknown'}, {'state':'holding'}, {'state':'empty'}):
            with self.assertRaises(ValueError):
                startup_decision(record, 'holding', True)
        with self.assertRaises(ValueError):
            startup_decision({'state':'holding','held_command':26,'opening':30}, 'empty', True)

    def setup_execution(self, bus):
        for name,value in [('imshow',None),('waitKey',-1),('imwrite',True)]:
            self.enterContext(patch.object(pp.cv2,name,return_value=value))
        self.enterContext(patch.object(pp.time,'sleep'))
        self.enterContext(patch.object(pp,'recovery_context',return_value='test_context'))
        robot=SimpleNamespace(bus=bus,configure=Mock())
        cap=SimpleNamespace(read=lambda:(True,np.zeros((8,8,3),np.uint8)))
        return robot,cap

    def test_crash_during_closing_leaves_unknown(self):
        bus=Bus(); robot,cap=self.setup_execution(bus); planner=FakePlanner()
        bus.write=Mock(side_effect=RuntimeError('communication lost'))
        self.store.write('empty')
        with self.assertRaisesRegex(RuntimeError,'communication lost'):
            pp.execute(robot,planner,stages(planner),observation(),cap,self.root,state_store=self.store)
        self.assertEqual(self.store.read()['state'],'unknown')
        self.assertEqual(self.store.read()['reason'],'closing_gripper')

    def test_loaded_failure_keeps_restart_record(self):
        bus=Bus(fail_lift=True); robot,cap=self.setup_execution(bus); planner=FakePlanner()
        with self.assertRaises(pp.MotionFailure):
            pp.execute(robot,planner,stages(planner),observation(),cap,self.root,state_store=self.store)
        record=self.store.read()
        self.assertEqual(record['state'],'holding')
        self.assertEqual(record['held_command'],bus.commands[-1]['gripper'])
        self.assertEqual(record['context'],'test_context')

    def test_recovery_never_reconfigures_or_opens_before_release(self):
        bus=Bus(); bus.q['gripper']=30
        robot,cap=self.setup_execution(bus); planner=FakePlanner()
        plan=[s for s in stages(planner) if s['stage'] in ('lift','transfer','release','retract')]
        obs=observation(); obs['gripper']=30
        recovery=dict(state='holding',held_command=26,opening=30)
        self.store.write(**recovery)
        pp.execute(robot,planner,plan,obs,cap,self.root,state_store=self.store,recovery=recovery)
        robot.configure.assert_not_called()
        first_open=next(i for i,c in enumerate(bus.commands) if c.get('gripper',0)>26)
        self.assertTrue(all(c['gripper']==26 for c in bus.commands[:first_open]))
        self.assertEqual(bus.commands[first_open]['wrist_roll'],10)
        self.assertEqual(self.store.read()['state'],'unknown')
        self.assertIn('release',self.store.read()['reason'])

    def test_auto_startup_unknown_never_connects_robot(self):
        # An interrupted operation must still block even with the initial-empty default.
        self.store.write('unknown', reason='closing_gripper')
        c=json.loads((pp.ROOT/'pick_place_config.json').read_text())
        c['robot_id']='test'
        config=self.root/'config.json'; config.write_text(json.dumps(c))
        # Release our test lock so main owns it for this check.
        self.store.close()
        with patch('sys.argv',['pick_place.py','--execute','--pick-only','--auto','--config',str(config),'--output',str(self.root/'different_output')]), \
             patch.object(pp,'GraspState',side_effect=lambda c:GraspState(c,self.root/'state')), \
             patch.object(pp,'Scene'), patch.object(pp,'robot_reader') as connect, \
             patch.object(pp.cv2,'destroyAllWindows'):
            with self.assertRaisesRegex(ValueError,'Startup state unknown'):
                pp.main()
            connect.assert_not_called()

    def test_default_empty_and_explicit_overrides(self):
        self.store.write('empty')
        self.assertEqual(startup_decision(self.store.read(), None), 'empty')
        with self.assertRaises(ValueError):
            startup_decision(self.store.read(), 'unknown')
        self.store.write('holding', held_command=26, opening=30)
        with self.assertRaises(ValueError):
            startup_decision(self.store.read(), None)
        with self.assertRaises(ValueError):
            startup_decision({'state':'empty'}, None, recover=True)

    def test_first_auto_start_reaches_robot_without_state_prompt(self):
        c=json.loads((pp.ROOT/'pick_place_config.json').read_text()); c['robot_id']='test'
        config=self.root/'config.json'; config.write_text(json.dumps(c))
        self.store.close()
        with patch('sys.argv',['pick_place.py','--execute','--pick-only','--auto','--config',str(config),'--output',str(self.root/'output')]), \
             patch.object(pp,'GraspState',side_effect=lambda c:GraspState(c,self.root/'state')), \
             patch.object(pp,'Scene'), patch.object(pp,'robot_reader',side_effect=RuntimeError('mock connection reached')) as connect, \
             patch('builtins.input') as prompt, patch.object(pp.cv2,'destroyAllWindows'):
            with self.assertRaisesRegex(RuntimeError,'mock connection reached'):
                pp.main()
            connect.assert_called_once()
            prompt.assert_not_called()

    def test_recovery_feedback_is_read_only_and_rejects_loss_of_grip_or_torque(self):
        planner = FakePlanner()
        recovery = dict(held_command=26, opening=30)
        for opening, torque, valid in [(30, 1, True), (0, 1, False), (40, 1, False), (30, 0, False)]:
            bus = Mock()
            bus.sync_read.side_effect = lambda register: ({'wrist_roll': 0., 'gripper': opening}
                if register == 'Present_Position' else {'wrist_roll': torque, 'gripper': torque})
            if valid:
                pp.validate_recovery_feedback(SimpleNamespace(bus=bus), planner.fk, planner.c, recovery)
            else:
                with self.assertRaises(ValueError):
                    pp.validate_recovery_feedback(SimpleNamespace(bus=bus), planner.fk, planner.c, recovery)
            bus.sync_write.assert_not_called()

    def test_recovery_requires_only_tray_tag_not_cube_id(self):
        c = json.loads((pp.ROOT/'pick_place_config.json').read_text())
        c.update(recover_place=True, detector='aruco', pick_only=False)
        c['cube']['id'] = None
        c['plate'].update(id=3, marker_mm=18, rim_height_mm=5,
                          inner_half_size_mm=[45,45], outer_half_size_mm=[50,50])
        pp.check_config(c)
        self.assertEqual(pp.roles(c), ('plate',))

    def test_recovery_plan_only_carries_and_releases(self):
        planner=FakePlanner()
        planner.c.update(tcp_lowest_extent_mm=10,release_gap_mm=3)
        planner.c['cube']['grasp_depth_mm']=12
        planner.c['plate']=dict(drop_offset_mm=[0,0,0],rim_height_mm=5)
        obs=observation(); obs['joints_deg']['wrist_roll']=20
        plan=pp.recovery_plan(planner,obs)
        self.assertEqual([s['stage'] for s in plan],['lift','transfer','place','release','retract'])
        self.assertGreaterEqual(plan[0]['T_base_tcp'][2][3],.12)
        self.assertAlmostEqual(plan[2]['T_base_tcp'][2][3],.016)


if __name__=='__main__':
    unittest.main()
