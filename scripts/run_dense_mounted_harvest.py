#!/usr/bin/env python3
"""Pre-SDK dry run or explicitly authorized, one-shot dense mounted diagnostic.

Default is --dry-run. Execution needs a separate exact signed-off JSON; no old
smoke/pilot permission is consumed. The supervisor only terminates its own PGID.
"""
import argparse,dataclasses,hashlib,importlib,json,os,signal,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT/'harvest_rl/harvest_rl'
p=types.ModuleType('_dense_review');p.__path__=[str(PKG)];sys.modules[p.__name__]=p
R=importlib.import_module('_dense_review.dense_runtime_review')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--dry-run',action='store_true')
parser.add_argument('--execute',action='store_true')
parser.add_argument('--authorization',type=Path)
parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
a=parser.parse_args()
if a.dry_run and a.execute:parser.error('Choose dry-run or execute')
review,manifest=R.load_review(a.review,authorization=a.authorization if a.execute else None)
if a.execute and not a.authorization:parser.error('--execute requires exact reviewed --authorization')
if not a.execute:
 print(json.dumps(dict(pre_sdk=True,sdk_imported=False,runtime_authorized=False,source_and_asset_hashes_match=True,
  physical_fruit_count=len(manifest['inventory']),target_order=review['target_order'],feasible_ids=review['feasible_ids'],
  actual_initialized_state=None,bounds=review['bounds']),indent=2));raise SystemExit(0)
if not a.authorization:parser.error('--execute requires exact reviewed --authorization')
auth=json.loads(a.authorization.read_text())
context=R.runtime_context(review,a.review,authorization=auth)
output=context['output']
if not a.worker:
 competitors=R.competing_trials()
 if competitors:raise SystemExit('Competing trials: '+json.dumps(competitors))
 output.mkdir(parents=True,exist_ok=False)
 claim=dict(status='running',attempts=context['prior_attempts']+1,failures=context['prior_failures'],current_attempt=context['prior_attempts']+1,current_attempt_failures=0,continuation_context={k:str(v) for k,v in context.items() if k!='output'},review_sha256=R.sha(a.review),authorization=auth,parent_history=review['history'],started_unix=time.time(),automatic_retry=False)
 ledger=output/'ledger.json';ledger.write_text(json.dumps(claim,indent=2))
 command=[sys.executable,'-B',str(Path(__file__).resolve()),'--review',str(a.review.resolve()),'--execute','--authorization',str(a.authorization.resolve()),'--worker']
 with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
  child=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,env={**os.environ,'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
  timeout=False
  try:code=child.wait(timeout=review['bounds']['wall_seconds'])
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=review['bounds']['cleanup_seconds'])
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
 passed=code==0 and not timeout and (result.get('startup_capture_complete') is True and result.get('initial_inventory_stable') is True if review.get('startup_capture_only') is True else result.get('diagnostic_targets_complete') is True)
 status=('startup_capture_complete_pending_review' if review.get('startup_capture_only') else 'completed_pending_review') if passed else 'failed_stop_no_retry'
 claim.update(status=status,failures=context['prior_failures']+(0 if passed else 1),current_attempt_failures=0 if passed else 1,process_exit=code,wall_limit=timeout,elapsed_seconds=time.time()-claim['started_unix'])
 ledger.write_text(json.dumps(claim,indent=2));print(json.dumps(claim,indent=2));raise SystemExit(0 if passed else 1)
# Only the reviewed one-shot child reaches this SDK boundary.
if not output.is_dir() or not (output/'ledger.json').is_file():raise SystemExit('Missing supervisor claim')
with (output/'worker_claim.json').open('x') as f:
 json.dump(dict(pid=os.getpid(),started_unix=time.time(),no_retry=True),f)
os.nice(19)
if hasattr(os,'sched_setaffinity'):
 available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[-2:])
sys.path.insert(0,str(ROOT/'harvest_rl'))
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device='cpu',enable_cameras=False,kit_args=f'--portable-root {output}/kit --/log/file={output}/kit.log')
app=launcher.app
env=None;native=None;window_evidence=None;result=dict(diagnostic_targets_complete=False,named_tree_success=False,all_reachable_claim=False,scope='unvalidated_dense_mounted_diagnostic',error=None)
class StartupCaptureFinished(Exception):pass
def serial(value):
 if dataclasses.is_dataclass(value):return dataclasses.asdict(value)
 if hasattr(value,'detach'):return value.detach().cpu().tolist()
 if isinstance(value,(set,frozenset)):return sorted(value)
 if isinstance(value,Path):return str(value)
 raise TypeError(type(value).__name__)
try:
 import torch
 from harvest_rl.pluck.env_cfg import PluckEnvCfg
 from harvest_rl.tree import Fruit
 from harvest_rl.orchard_dense_env_cfg import configure_dense_inventory
 from harvest_rl.orchard_mounted_basket import MountedBasketSpec,choose_empty_deposit_slot
 from harvest_rl.orchard_mounted_env import configure_dense_mounted_case,make_dense_mounted_env_class
 from harvest_rl.mounted_native_telemetry import NativeMountedTelemetryReader
 from harvest_rl.dense_window_evidence import DurableNativeWindows,startup_capture_summary
 from harvest_rl.mounted_basket_verifier import BasketGeometry,FruitGeometry,MountedBasketVerifier
 from harvest_rl.harvest_cycle import BasketProposal,FruitTarget,HarvestCycle
 from harvest_rl.harvest_cycle_adapter import ParkedTaskSignals
 from harvest_rl.dense_cycle_adapter import DenseHarvestCycleAdapter
 from harvest_rl.baseline.pick_place_env import PickPlaceEnv
 from harvest_rl.checkpoint_policy import CheckpointPolicy
 torch.manual_seed(42)
 cfg=PluckEnvCfg();cfg.scene.num_envs=1;cfg.sim.device='cpu';cfg.cameras=False
 cfg.scene.apple.spawn.usd_path=manifest['apple_usd']
 # Frozen B needs the unchanged authentic branch geometry at handoff.
 cfg.case_branches=json.loads(Path(manifest['source_tree_json']).read_text())['branches']
 from harvest_rl.compat import backend_quat_xyzw,canonical_quat
 from harvest_rl.dense_stance import bind_reviewed_stance,verify_measured_root
 stance=review['robot_stance']
 bind_reviewed_stance(cfg,stance,backend_quat_xyzw)
 cfg=configure_dense_inventory(cfg,manifest['inventory'],manifest['tree_usd'],active_fruit_id=review['target_order'][0])
 spec=MountedBasketSpec(**review['mount'])
 window_evidence=DurableNativeWindows(output/'raw_contacts.jsonl')
 native=NativeMountedTelemetryReader(window_sink=window_evidence)
 contact={**review['contact_review'],'runtime_review_reference':auth['review_reference']}
 cfg=configure_dense_mounted_case(cfg,spec,feasible_target_ids=review['feasible_ids'],diagnostic_target_ids=() if review['feasible_ids'] else review['target_order'],contact_reader=native,safety_reader=native.safety,
  prior_failures=review['history']['native_reset_historical_failures'],runtime_review_reference=auth['review_reference'],contact_review=contact,
  history_review_reference=auth['review_reference'],max_task_seconds=review['bounds']['task_seconds'],
  camera_joint_filters=review.get('camera_joint_filters',False))
 env=make_dense_mounted_env_class()(cfg=cfg)
 env.reset(seed=42)
 def initialization_snapshot():
  return dict(actual_robot_joint_pos=env._robot.data.joint_pos,actual_actions=env.actions,
   root_position_m=env._robot.data.root_pos_w[0].detach().cpu().tolist(),
   root_orientation_xyzw=canonical_quat(env._robot.data.root_quat_w)[0].detach().cpu().tolist(),
   inventory=env.inventory_states(),basket=env._mounted_motion(float(env.episode_length_buf[0])*env.step_dt),native_identity_map=native.geometry,
   camera_joint_filter_proof=getattr(env,'_camera_joint_filter_proof',None))
 before=initialization_snapshot()
 (output/'before_prime.json').write_text(json.dumps(before,default=serial,indent=2))
 verify_measured_root(before['root_position_m'],before['root_orientation_xyzw'],stance)
 try:env.prime_native_telemetry()
 except BaseException:
  try:(output/'actual_initialization.json').write_text(json.dumps(initialization_snapshot(),default=serial,indent=2))
  except BaseException as capture_error:result['initialization_capture_error']=repr(capture_error)
  raise
 initial=initialization_snapshot()
 (output/'actual_initialization.json').write_text(json.dumps(initial,default=serial,indent=2))
 verify_measured_root(initial['root_position_m'],initial['root_orientation_xyzw'],stance)
 if review.get('startup_capture_only') is True:
  detached=[i for i,s in initial['inventory'].items() if not bool(s['apple_attached'][0])]
  result.update(startup_capture_summary(native.window,records=window_evidence.records,control_steps=int(env.episode_length_buf[0]),actions=env.actions[0].detach().cpu().tolist(),detached_ids=detached))
  raise StartupCaptureFinished()
 allrows={x['target_id']:x for x in manifest['inventory']}
 geometry=BasketGeometry(env._mounted_plan['body_path'],env._mounted_plan['contact_ids']['floor_contact_id'],tuple(env._mounted_plan['contact_ids']['wall_contact_ids']),tuple(spec.inner_size_m))
 fruits=env.dense_fruit_inventory()
 verifier=MountedBasketVerifier(geometry,[FruitGeometry(x['target_id'],x['body_id'],tuple(x['half_extents_m']),x['mass_kg']) for x in fruits],
  hand_body_ids=native.hand_actor_ids,other_body_ids=tuple(x for x in native.other_actor_ids if x not in {f['body_id'] for f in fruits} and x not in native.hand_actor_ids and x not in {geometry.floor_contact_id,*geometry.wall_contact_ids}))
 targets=[FruitTarget(i,allrows[i]['tree_id'],allrows[i]['fruit_id'],allrows[i]['fruit']['mass'],allrows[i]['parent_branch_id'],review['manifest_sha256']) for i in review['target_order']]
 cycle=HarvestCycle(targets,basket=BasketProposal(inner_size_m=tuple(spec.inner_size_m),dry_mass_kg=spec.dry_mass_kg,payload_limit_kg=review['payload_limit_kg']),prior_failures=0)
 def read(e):
  frame,safety=e.read_mounted_telemetry();s=e.get_state()
  flag=lambda k:bool(s[k][0]);norm=lambda x:float(torch.linalg.norm(x[0]))
  # Unscheduled fruit must not be knocked off or silently disappear.
  collateral=[f['target_id'] for f in e.dense_fruit_inventory() if f['target_id'] not in cycle.deposited and f['target_id']!=cycle.active and not f['attached']]
  if collateral:raise RuntimeError('Collateral fruit detached: '+','.join(collateral))
  return frame,ParkedTaskSignals(frame.sample,frame.time_s,str(e.active_target_id),flag('apple_attached'),flag('grasp_contact'),flag('opposing_contact'),flag('carried'),flag('released'),int(s['failure'][0]),
   extra_apple_assistance=norm(e.apple_external_force)>0,fallen=safety['fallen'],unsafe_collision=safety['unsafe_collision'],base_linear_speed_m_s=norm(e._robot.data.root_lin_vel_w),base_angular_speed_rad_s=norm(e._robot.data.root_ang_vel_w),time_limit_reached=e.dense_timeout_reached)
 def slot(frame,verified,target):
  centers=[f.position_b_m for f in verified.fruits if f.target_id in cycle.deposited]
  return choose_empty_deposit_slot(cfg.dense_capacity_plan,centers,tuple(allrows[target]['fruit']['half_extents']))
 policy=CheckpointPolicy(review['checkpoint']['path'],'cpu',normalizer_epsilon=.01)
 adapter=DenseHarvestCycleAdapter(env,policy,cycle,verifier,target_order=review['target_order'],reader=read,observation_builder=PickPlaceEnv._get_observations,slot_selector=slot,max_pick_steps=review['bounds']['per_pick_steps'],max_stow_steps=review['bounds']['stow_steps'])
 with (output/'trace.jsonl').open('x') as trace:
  for _ in range(review['bounds']['control_steps']-1):
   action=adapter.action()
   trace.write(json.dumps(dict(sample=adapter.last_sample,cycle=cycle.snapshot(),verification=adapter.last_result,inventory=env.inventory_states()),default=serial)+'\n');trace.flush()
   if action is None:break
   env.step(action)
  result.update(diagnostic_targets_complete=adapter.named_tree_success,named_tree_success=adapter.named_tree_success and not review['unresolved_ids'],
   cycle=cycle.snapshot(),two_pick_checkpoint=adapter.two_pick_checkpoint,final_retained_ids=sorted(adapter.last_result.retained_ids) if adapter.last_result else [],
   named_feasible_subset_success=adapter.named_tree_success and bool(review['feasible_ids']),
   contact_mapping_verified=getattr(native,'contact_mapping_verified',False),mapping_observations=getattr(native,'mapping_observations',[]),world_resets_after_start=max(0,env._dense_reset_count-1),explicit_initial_resets=env._dense_reset_count)
except StartupCaptureFinished:
 pass
except BaseException as exc:
 result['error']=f'{type(exc).__name__}: {exc}'
finally:
 # Persist evidence before native close; post-close errors cannot fabricate pass.
 if native is not None:
  result['native_contact_errors']=list(native.buffer.errors) if native.buffer is not None else []
 if window_evidence is not None:result['raw_window_records']=window_evidence.records
 try:
  (output/'result.json').write_text(json.dumps(result,default=serial,indent=2)+'\n')
  if window_evidence is not None:window_evidence.close()
 finally:
  try:
   if env is not None:env.close()
  finally:app.close()
raise SystemExit(0 if (result['diagnostic_targets_complete'] or result.get('startup_capture_complete')) and not result['error'] else 1)
