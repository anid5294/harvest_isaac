"""Small pre-SDK packet checker for the opt-in dense mounted diagnostic."""
import hashlib,json,os
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_review(path, *, authorization=None):
 path=Path(path).resolve();r=json.loads(path.read_text());root=Path(r['repository'])
 if r.get('schema')!='dense_mounted_runtime_review_v1' or r['runtime_authorized'] is not False:
  raise ValueError('Require the frozen unapproved review packet')
 for name,expected in r['source_hashes'].items():
  if sha(root/name)!=expected:raise ValueError('Changed source: '+name)
 manifest_path=Path(r['manifest']);m=json.loads(manifest_path.read_text())
 if sha(manifest_path)!=r['manifest_sha256']:raise ValueError('Changed full inventory manifest')
 for name,expected in m['files'].items():
  if sha(manifest_path.parent/name)!=expected:raise ValueError('Changed dense asset: '+name)
 if sha(r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Changed checkpoint')
 for name,expected in r.get('asset_hashes',{}).items():
  if sha(name)!=expected:raise ValueError('Changed native asset/model: '+name)
 if r.get('mount_screen'):
  ref=r['mount_screen']
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed mount screen')
 for ref in r['preserved_evidence']:
  if sha(root/ref['path'])!=ref['sha256']:raise ValueError('Changed parent evidence: '+ref['path'])
 ids={str(x['target_id']) for x in m['inventory']}
 order=r['target_order']
 if len(order)<3 or len(set(order))!=len(order) or not set(order)<=ids:
  raise ValueError('Need three distinct frozen in-scene diagnostic targets')
 if len(m['inventory'])!=r['physical_fruit_count'] or len(ids)!=len(m['inventory']):
  raise ValueError('Full physical inventory mismatch')
 b=r['bounds']
 if (b['episodes']!=1 or b['worlds']!=1 or b['cameras'] or b['training_updates']!=0
   or b['cleanup_seconds']!=15 or b['task_seconds']>40*len(order)+5
   or b['wall_seconds']>900 or b['control_steps']>30*b['task_seconds']+1):
  raise ValueError('Changed reviewed runtime bounds')
 if authorization is not None:
  a=json.loads(Path(authorization).read_text())
  if (a.get('approved') is not True or not a.get('review_reference')
    or a.get('packet_sha256')!=sha(path) or a.get('history_reviewed') is not True
    or a.get('allow_unresolved_diagnostic_targets') is not True):
   raise ValueError('Explicit exact review/history/unresolved diagnostic approval required')
  if r.get('startup_capture_only') and a.get('human_review_clearance_after_three') is not True:
   raise ValueError('Startup capture requires explicit human review clearance of the3/3stop')
 stop=r.get('stopped_dense_group_ref')
 if stop is not None:
  if sha(stop['path'])!=stop['sha256']:raise ValueError('Changed dense3/3stop record')
  record=json.loads(Path(stop['path']).read_text())
  if (record.get('attempts'),record.get('failures'))!=(3,3):raise ValueError('Stopped dense history must be preserved')
  if authorization is not None and not r.get('continuation'):
   raise ValueError('New source layout cannot reset3/3history; require explicit reviewed same-group continuation')
 if stop is None or r.get('continuation') or authorization is not None:
  runtime_context(r,path,authorization=a if authorization is not None else None)
 return r,m

def runtime_context(review, review_path, *, authorization=None):
 """Explicit reviewed continuations; each exception binds immutable history."""
 c=review.get('continuation')
 if c is None:
  return dict(output=Path(review_path).resolve().parent/'runtime_once',prior_attempts=0,prior_failures=0)
 attempt=c.get('attempt_number')
 if attempt==6 and c.get('schema')=='dense_mounted_hand_island_cleared_attempt6_v1':
  return _hand_island_cleared_attempt6(review,authorization)
 if attempt==5 and c.get('schema')=='dense_mounted_camera_cleared_attempt5_v1':
  return _camera_cleared_attempt5(review,authorization)
 if attempt==4 and c.get('schema')=='dense_mounted_human_cleared_attempt4_v1':
  return _human_cleared_attempt4(review,authorization)
 if attempt not in (2,3) or c.get('schema')!=f'dense_mounted_reviewed_attempt{attempt}_v1':
  raise ValueError('Only explicitly reviewed second/third attempts are supported; no automatic retry')
 parent_path=Path(c['parent_ledger']['path']).resolve()
 parent_review_path=Path(c['parent_review']['path']).resolve()
 if sha(parent_path)!=c['parent_ledger']['sha256'] or sha(parent_review_path)!=c['parent_review']['sha256']:
  raise ValueError('Changed continuation parent history')
 parent=json.loads(parent_path.read_text());previous=json.loads(parent_review_path.read_text())
 group=Path(c.get('group_root','')).resolve()
 expected_parent=group/'ledger.json' if attempt==2 else group/'continuations'/'attempt002'/'ledger.json'
 if (parent.get('status')!='failed_stop_no_retry' or parent.get('attempts')!=attempt-1
     or parent.get('failures')!=attempt-1 or parent.get('review_sha256')!=sha(parent_review_path)
     or parent_path!=expected_parent):
  raise ValueError('Continuation requires the exact stopped cumulative parent group')
 if attempt==3:
  initial=group/'ledger.json'
  if (sha(initial)!=parent.get('continuation_context',{}).get('parent_ledger_sha256')
      or c.get('reviewed_failure_ledger_sha256s')!=[sha(initial),sha(parent_path)]):
   raise ValueError('Third attempt requires review of both immutable failed ledgers')
  if authorization is not None and authorization.get('reviewed_failure_ledger_sha256s')!=c['reviewed_failure_ledger_sha256s']:
   raise ValueError('Third attempt authorization must bind both failures; third failure requires human inspection')
 for field in ('bounds','target_order','mount','robot_stance','physical_fruit_count','history'):
  if review[field]!=previous[field]:raise ValueError('Continuation changed physical scope/budget/history: '+field)
 if review['checkpoint']['sha256']!=previous['checkpoint']['sha256']:
  raise ValueError('Continuation changed policy checkpoint')
 if sha(previous['manifest'])!=previous['manifest_sha256']:
  raise ValueError('Changed parent physical inventory')
 if json.loads(Path(review['manifest']).read_text())['inventory']!=json.loads(Path(previous['manifest']).read_text())['inventory']:
  raise ValueError('Continuation changed fruit inventory/eligibility')
 if authorization is not None and authorization.get('continuation_parent_ledger_sha256')!=sha(parent_path):
  raise ValueError('Continuation authorization must explicitly bind the failed parent ledger')
 return dict(output=group/'continuations'/f'attempt{attempt:03d}',prior_attempts=attempt-1,prior_failures=attempt-1,
             parent_ledger_sha256=sha(parent_path),parent_ledger=str(parent_path))

def _human_cleared_attempt4(review, authorization):
 """Single Nov3 user-cleared integrated diagnostic, never a generic retry."""
 c=review['continuation'];group=Path(c['group_root']).resolve()
 def hashed(ref):
  if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed human-reviewed history/proposal')
  return json.loads(Path(ref['path']).read_text())
 proposal=hashed(c['proposal']);patch=hashed(c['patch_supplement'])
 if (proposal.get('schema')!='dense_integrated_human_review_proposal_v1'
     or patch.get('proposal_sha256')!=c['proposal']['sha256']):
  raise ValueError('Require reviewed integrated proposal and stance patch')
 history=proposal['parent_history'];ledgers=c['reviewed_failure_ledgers']
 expected_paths=[group/'ledger.json']+[group/'continuations'/f'attempt{i:03d}'/'ledger.json' for i in (2,3)]
 if len(ledgers)!=3:raise ValueError('Human clearance requires all three failures')
 for i,(ref,frozen,path) in enumerate(zip(ledgers,history['ledgers'],expected_paths),1):
  ledger=hashed(ref)
  if (Path(ref['path']).resolve()!=path or ref['sha256']!=frozen['sha256']
      or (ledger.get('attempts'),ledger.get('failures'),ledger.get('status'))!=(i,i,'failed_stop_no_retry')
      or ledger.get('review_sha256')!=frozen['bound_review_sha256']):
   raise ValueError('Changed original cumulative failed group')
 previous=hashed(c['parent_review'])
 if c['parent_review']['sha256']!=history['ledgers'][-1]['bound_review_sha256'] or review['history']!=previous['history']:
  raise ValueError('Changed parent review/historical failure counters')
 stop=hashed(review['stopped_dense_group_ref']);held=hashed(c['prior_user_hold'])
 if (review['stopped_dense_group_ref']['sha256']!=history['stopped_group_ref']['sha256']
     or c['prior_user_hold']['sha256']!=history['hold_ref']['sha256']
     or (stop.get('attempts'),stop.get('failures'))!=(3,3)):
  raise ValueError('Human clearance cannot reset stopped history')
 for field in ('bounds','robot_stance','target_order','mount','physical_fruit_count','payload_limit_kg','feasible_ids'):
  if review[field]!=proposal[field]:raise ValueError('Changed approved diagnostic scope: '+field)
 if review['checkpoint']['sha256']!=proposal['checkpoint']['sha256'] or review['contact_review']!=proposal['contact_criteria']:
  raise ValueError('Changed approved policy/contact contract')
 frozen_manifest=hashed(c['frozen_manifest']);manifest=json.loads(Path(review['manifest']).read_text())
 if (c['frozen_manifest']['sha256']!=proposal['manifest_sha256']
     or any(manifest[key]!=frozen_manifest[key] for key in ('inventory','files','tree_id'))):
  raise ValueError('Changed reviewed corrected full-fruit geometry')
 expected_sources=dict(patch['source_hashes']);expected_sources['harvest_rl/harvest_rl/dense_runtime_review.py']=sha(__file__)
 if review['source_hashes']!=expected_sources:raise ValueError('Unreviewed source changes beyond narrow continuation')
 if authorization is not None:
  if (authorization.get('human_review_clearance_after_three') is not True
      or authorization.get('proposal_sha256')!=c['proposal']['sha256']
      or authorization.get('patch_supplement_sha256')!=c['patch_supplement']['sha256']
      or authorization.get('reviewed_failure_ledger_sha256s')!=[r['sha256'] for r in ledgers]
      or authorization.get('continuation_parent_ledger_sha256')!=ledgers[-1]['sha256']):
   raise ValueError('Explicit user clearance must bind proposal/patch/all3failures')
 return dict(output=group/'continuations'/'attempt004',prior_attempts=3,prior_failures=3,
  parent_ledger_sha256=ledgers[-1]['sha256'],parent_ledger=ledgers[-1]['path'],
  human_review_clearance=True,proposal_sha256=c['proposal']['sha256'])

def _camera_cleared_attempt5(review, authorization):
 """One explicitly approved camera-adjacency validation; no sixth attempt."""
 import copy
 c=review['continuation'];ref=c['camera_proposal']
 if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed camera review proposal')
 proposal=json.loads(Path(ref['path']).read_text())
 if proposal.get('schema')!='dense_camera_adjacency_validation_proposal_v1':
  raise ValueError('Require exact camera-adjacency proposal')
 expected=dict(proposal['source_hashes']);expected['harvest_rl/harvest_rl/dense_runtime_review.py']=sha(__file__)
 if review['source_hashes']!=expected or review.get('camera_joint_filters') is not True:
  raise ValueError('Changed approved camera correction/source closure')
 # Recheck the original three immutable failures and unchanged physical scope
 # through the prior validator. Actual current source closure was checked above.
 inherited=copy.deepcopy(review)
 patch=json.loads(Path(c['patch_supplement']['path']).read_text())
 inherited['source_hashes']=dict(patch['source_hashes'])
 inherited['source_hashes']['harvest_rl/harvest_rl/dense_runtime_review.py']=sha(__file__)
 context=_human_cleared_attempt4(inherited,None)
 group=Path(c['group_root']).resolve();fourth=c['fourth_ledger'];previous=c['fourth_parent_review'];stop=c['fourth_stop']
 for row,expected_ref in ((fourth,proposal['parent_ledger']),(stop,proposal['parent_stop'])):
  if sha(row['path'])!=row['sha256'] or row['sha256']!=expected_ref['sha256']:
   raise ValueError('Changed4/4camera parent history')
 if sha(previous['path'])!=previous['sha256']:raise ValueError('Changed fourth parent review')
 ledger=json.loads(Path(fourth['path']).read_text());marker=json.loads(Path(stop['path']).read_text());old=json.loads(Path(previous['path']).read_text())
 if (Path(fourth['path']).resolve()!=group/'continuations/attempt004/ledger.json'
     or (ledger.get('attempts'),ledger.get('failures'),ledger.get('status'))!=(4,4,'failed_stop_no_retry')
     or ledger.get('review_sha256')!=previous['sha256']
     or (marker.get('attempts'),marker.get('failures'))!=(4,4)
     or marker.get('fourth_ledger_sha256')!=fourth['sha256'] or old['history']!=review['history']):
  raise ValueError('Camera continuation must retain original4/4group')
 for field in ('bounds','robot_stance','target_order','mount','physical_fruit_count','payload_limit_kg','feasible_ids'):
  if review[field]!=old[field]:raise ValueError('Camera correction changed diagnostic scope')
 if authorization is not None:
  hashes=[r['sha256'] for r in c['reviewed_failure_ledgers']]+[fourth['sha256']]
  if (authorization.get('human_review_clearance_after_four') is not True
      or authorization.get('camera_proposal_sha256')!=ref['sha256']
      or authorization.get('reviewed_failure_ledger_sha256s')!=hashes
      or authorization.get('continuation_parent_ledger_sha256')!=fourth['sha256']):
   raise ValueError('Explicit camera correction/all4failures clearance required')
 return dict(output=group/'continuations'/'attempt005',prior_attempts=4,prior_failures=4,
  parent_ledger_sha256=fourth['sha256'],parent_ledger=fourth['path'],
  human_review_clearance=True,camera_proposal_sha256=ref['sha256'])

def _hand_island_cleared_attempt6(review, authorization):
 """Single user-approved complete hand-assembly correction; no seventh retry."""
 import copy
 c=review['continuation'];ref=c['hand_island_proposal']
 if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed hand-island review proposal')
 proposal=json.loads(Path(ref['path']).read_text())
 if proposal.get('schema')!='dense_hand_island_validation_proposal_v2' or len(proposal['exact_pair_proof']['pairs'])!=18:
  raise ValueError('Require exact complete18pair hand-island proposal')
 expected=dict(proposal['source_hashes']);expected['harvest_rl/harvest_rl/dense_runtime_review.py']=sha(__file__)
 if review['source_hashes']!=expected or review.get('camera_joint_filters') is not True:
  raise ValueError('Changed reviewed hand-island correction/source closure')
 # Revalidate the previous four-failure lineage and original scope. Physical
 # source closure for this correction is separately verified above/load_review.
 inherited=copy.deepcopy(review)
 original=json.loads(Path(c['camera_proposal']['path']).read_text())
 inherited['source_hashes']=dict(original['source_hashes'])
 inherited['source_hashes']['harvest_rl/harvest_rl/dense_runtime_review.py']=sha(__file__)
 _camera_cleared_attempt5(inherited,None)
 group=Path(c['group_root']).resolve();fifth=c['fifth_ledger'];stop=c['fifth_stop'];previous=c['fifth_parent_review']
 for row,frozen in ((fifth,proposal['parent_ledger']),(stop,proposal['parent_stop'])):
  if sha(row['path'])!=row['sha256'] or row['sha256']!=frozen['sha256']:
   raise ValueError('Changed5/5hand-island parent history')
 if sha(previous['path'])!=previous['sha256']:raise ValueError('Changed fifth parent review')
 ledger=json.loads(Path(fifth['path']).read_text());marker=json.loads(Path(stop['path']).read_text());old=json.loads(Path(previous['path']).read_text())
 if (Path(fifth['path']).resolve()!=group/'continuations/attempt005/ledger.json'
     or (ledger.get('attempts'),ledger.get('failures'),ledger.get('status'))!=(5,5,'failed_stop_no_retry')
     or ledger.get('review_sha256')!=previous['sha256']
     or (marker.get('attempts'),marker.get('failures'))!=(5,5)
     or marker.get('fifth_ledger_sha256')!=fifth['sha256'] or old['history']!=review['history']):
  raise ValueError('Hand-island continuation must retain original5/5group')
 for field in ('bounds','robot_stance','target_order','mount','physical_fruit_count','payload_limit_kg','feasible_ids'):
  if review[field]!=old[field]:raise ValueError('Hand-island correction changed diagnostic scope')
 if authorization is not None:
  hashes=[r['sha256'] for r in c['reviewed_failure_ledgers']]+[c['fourth_ledger']['sha256'],fifth['sha256']]
  if (authorization.get('human_review_clearance_after_five') is not True
      or authorization.get('hand_island_proposal_sha256')!=ref['sha256']
      or authorization.get('reviewed_failure_ledger_sha256s')!=hashes
      or authorization.get('continuation_parent_ledger_sha256')!=fifth['sha256']):
   raise ValueError('Explicit full hand-island/all5failures clearance required')
 return dict(output=group/'continuations'/'attempt006',prior_attempts=5,prior_failures=5,
  parent_ledger_sha256=fifth['sha256'],parent_ledger=fifth['path'],
  human_review_clearance=True,hand_island_proposal_sha256=ref['sha256'])

def competing_trials():
 """Inspect process metadata only; no signals or other-project file reads."""
 import subprocess,re
 rows=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True)
 parsed=[]
 for line in rows.splitlines():
  fields=line.strip().split(None,2)
  if len(fields)==3:parsed.append((int(fields[0]),int(fields[1]),fields[2]))
 parents={p:q for p,q,_ in parsed};ancestors=set();pid=os.getpid()
 while pid and pid not in ancestors:ancestors.add(pid);pid=parents.get(pid,0)
 return [dict(pid=p,command=c) for p,_,c in parsed if p not in ancestors
  and re.search(r'(?:^|[ /])(?:run_env|arena_apple_pick|evaluate_harvest|train_acquisition_pilot|train_dense_acquisition_pilot|run_dense_mounted_harvest|validate_contact_filter_calibration)\.py(?:\s|$)|kit/kit|isaac-sim',c)]
