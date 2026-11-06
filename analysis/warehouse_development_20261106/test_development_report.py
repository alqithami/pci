"""Explicitly synthetic controls. Never imports or reads live evaluation outcomes."""
import copy,csv,hashlib,json,tempfile,sys
from pathlib import Path
import development_report as dr
ROOT=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/workspace/scratch/5bbfb44aa122/pci_work_20261106')

def put(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def make_fixture(root):
 sources,study,frontier=dr.load_protocol(SOURCE);c=root/'cohort';i=root/'inputs';e=c/'evaluation';c.mkdir();i.mkdir();e.mkdir()
 put(root/'SYNTHETIC_TEST_ONLY.json',{'label':'ARTIFICIAL SOFTWARE CONTROL; NOT SCIENTIFIC DATA'})
 protocol=study.protocol();put(i/'cases/PROTOCOL_DRAFT.json',protocol)
 configs={};descriptors=[];seeds=study.DEVELOPMENT_SEEDS;candidates=study.candidates()
 for seed in seeds:
  for candidate in candidates:
   cfg=study.train_config(candidate,seed,seeds.index(seed),role='development',steps=262144,lock_hash=dr.digest(protocol),domain='warehouse');run=cfg['run_id'];p=i/'cases'/(run+'.json');put(p,cfg);configs[run]=cfg
   descriptors.append({'case_id':run,'candidate_id':candidate['candidate_id'],'family':candidate['family'],'domain':'warehouse','n_agents':4,'training_seed':seed,'path':p.name,'sha256':dr.sha(p),'run_role':'development','total_steps':262144})
 ledger={'schema':'PCI-COMPLETE-CASE-MANIFEST-v1','stage':'development','case_count':144,'protocol_canonical_sha256':dr.digest(protocol),'cases':descriptors};put(i/'cases/CASE_MANIFEST.json',ledger)
 cfg={'schema':'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1','role':'scientific','stage':'development','study_domain':'warehouse','device':'cuda','worlds_per_profile':10,'out':'/synthetic/cohort','deadline_unix':1000,'source_design_lock':'/synthetic/inputs/SOURCE_DESIGN_LOCK.json','launch_lock':'/synthetic/inputs/LAUNCH_LOCK.json','readiness_receipt':'/synthetic/inputs/CUDA_READINESS.json','protocol_file':'/synthetic/inputs/cases/PROTOCOL_DRAFT.json'};put(i/'CONTROLLER_CONFIG.json',cfg)
 design={'schema':'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1','source_sha256':sources,'source_map_sha256':dr.digest(sources),'protocol_canonical_sha256':dr.digest(protocol),'scientific_launch_authorized':True,'engineering_complete':True};put(i/'SOURCE_DESIGN_LOCK.json',design)
 ready={'schema':'PCI-FAIR-CUDA-READINESS-v1','status':'PASS','all_required_methods_pass':True,'source_map_sha256':dr.digest(sources),'gpu_uuid':'SYNTHETIC','host_boot_id':'SYNTHETIC'};put(i/'CUDA_READINESS.json',ready)
 comparisons={'controller_config_sha256':dr.digest(cfg),'case_manifest_sha256':dr.sha(i/'cases/CASE_MANIFEST.json'),'source_design_lock_sha256':dr.sha(i/'SOURCE_DESIGN_LOCK.json'),'readiness_receipt_sha256':dr.sha(i/'CUDA_READINESS.json'),'source_map_sha256':dr.digest(sources),'out':cfg['out'],'deadline_unix':cfg['deadline_unix']}
 launch={'schema':'PCI-FAIR-SCIENTIFIC-LAUNCH-v1','scientific_launch_authorized':True,'gpu_uuid':'SYNTHETIC','host_boot_id':'SYNTHETIC',**comparisons};put(i/'LAUNCH_LOCK.json',launch)
 external={cfg[k]:dr.sha(i/name) for k,name in [('source_design_lock','SOURCE_DESIGN_LOCK.json'),('launch_lock','LAUNCH_LOCK.json'),('readiness_receipt','CUDA_READINESS.json'),('protocol_file','cases/PROTOCOL_DRAFT.json')]}
 ex={'controller_configuration':cfg,'controller_configuration_file_sha256':dr.sha(i/'CONTROLLER_CONFIG.json'),'source_sha256':sources,'case_manifest_sha256':comparisons['case_manifest_sha256'],'authority':{**comparisons,'external_record_sha256':external}};put(c/'EXECUTION_LOCK.json',ex)
 policies=[];cases=[]
 for row in descriptors:
  run=row['case_id'];v=configs[run];seed=row['training_seed'];cluster=seeds.index(seed)
  policies.append({k:row[k] for k in ('candidate_id','family','training_seed','domain','n_agents')}|{'directory':run,'run_id':run,'training_signature':'synthetic-signature-'+run,'portable_export_sha256':'synthetic-export-'+run})
  for profile in sorted(v['profiles']):
   for world in range(10):
    key=f'{run}__{profile}__w{world:02d}';ws=3200000000+4000000+1000*cluster+world
    cases.append({'case_id':key,'policy_directory':run,'candidate_id':row['candidate_id'],'family':row['family'],'method':v['method'],'training_seed':seed,'domain':'warehouse','n_agents':4,'profile_id':profile,'environment_config':v['profiles'][profile],'world_index':world,'world_seed':ws,'policy_seed':ws+50000000})
 eval_sources={n.removeprefix('runtime/'):v for n,v in sources.items() if n.startswith(('runtime/training/','runtime/evaluation_fair/'))}
 manifest={'schema':'PCI-FAIR-EVALUATION-MANIFEST-v1','policies':policies,'cases':cases,'training_case_manifest':ledger,'training_case_manifest_sha256':dr.digest(ledger),'stage':'development','run_role':'development','protocol_sha256':dr.digest(protocol),'source_sha256':eval_sources,'worlds_per_profile':10,'training_seed_clusters':seeds,'complete_policy_count':144,'complete_episode_count':5760}
 put(e/'MANIFEST.json',manifest);put(c/'EVALUATION_MANIFEST.json',manifest);manifest_hash=dr.digest(manifest)
 receipts={};intent={};reports=[];profile_rows=[];seed_rows=[]
 # Constant profile counts permit an independent closed-form aggregate.
 for index,policy in enumerate(policies):
  ci=index%48;seed=policy['training_seed'];physical=0 if policy['family']=='shieldaware' else 2;institution=0 if policy['family']=='shieldaware' else ci%10+1;delivered=1+ci%4
  for profile in sorted(dr.PROFILES):
   identity={k:policy[k] for k in ('candidate_id','family','training_seed','domain','n_agents')}
   counts={'physical_replacements':physical*10,'institutional_replacements':institution*10,'total_replacements':(physical+institution)*10,'deliveries':delivered*10,'pickups':delivered*10,'joint_steps':2560}
   profile_rows.append({**identity,'profile_id':profile,**counts})
  counts={k:4*counts[k] for k in dr.COUNTS};seed_rows.append({**identity,**counts,'cohort_complete':True,'evaluation_complete':True,'profile_weights':{p:.25 for p in sorted(dr.PROFILES)},'institutional_per_1000_agent_decisions':1000*counts['institutional_replacements']/(4*10240),'deliveries_per_1000_joint_steps':1000*counts['deliveries']/10240,'normalized_cost':counts['institutional_replacements']/(4*10240),'normalized_utility':counts['deliveries']/(4*10240)})
 by_run={p['run_id']:(j,p) for j,p in enumerate(policies)}
 for case in cases:
  index,policy=by_run[case['policy_directory']];ci=index%48;physical=0 if policy['family']=='shieldaware' else 2;institution=0 if policy['family']=='shieldaware' else ci%10+1;delivered=1+ci%4;key=case['case_id']
  totals={'physical_replacements':physical,'institutional_replacements':institution,'total_replacements':physical+institution,'deliveries':delivered,'pickups':delivered,'joint_steps':256}
  trace=hashlib.sha256(('synthetic-'+key).encode()).hexdigest();stamp=hashlib.sha256(('intent-'+key).encode()).hexdigest()
  body={'schema':'PCI-FAIR-EVALUATION-CASE-v1','status':'COMPLETE','case':case,'manifest_sha256':manifest_hash,'training_signature':policy['training_signature'],'portable_export_sha256':policy['portable_export_sha256'],'trace_seal':{'sha256':trace,'publication_intent_sha256':stamp,'records':258},'totals':totals,'deployment_policy_requested_rows':256*625 if case['family']=='shieldaware' else 0,'deployment_policy_callback_calls':256 if case['family']=='shieldaware' else 0}
  p=e/'cases'/(key+'.json');put(p,body);receipts[key]=dr.sha(p);intent[key]=stamp;reports.append({'case_id':key,'status':'PASS','trace_sha256':trace,'joint_steps':256,'agent_decisions':1024})
 analysis={'schema':'PCI-FAIR-EVALUATION-AGGREGATE-v1','status':'PASS_COMPLETE_COHORT','stage':'development','policy_count':144,'episode_count':5760,'equal_profile_seed_rows':seed_rows,'profile_seed_rows':profile_rows};put(e/'ANALYSIS.json',analysis);put(e/'development_selection_rows.json',seed_rows)
 complete={'schema':'PCI-FAIR-EVALUATION-COMPLETE-v1','status':'COMPLETE','manifest_sha256':manifest_hash,'analysis_sha256':dr.sha(e/'ANALYSIS.json'),'all_policy_count':144,'all_episode_count':5760,'joint_steps':1474560,'agent_decisions':5898240,'case_receipts':receipts,'trace_intent_receipts':intent};put(e/'EVALUATION_COMPLETE.json',complete)
 audit={'schema':'PCI-FAIR-RETAINED-RECORD-AUDIT-v1','status':'PASS','manifest_sha256':manifest_hash,'policy_count':144,'episode_count':5760,'joint_steps':1474560,'agent_decisions':5898240,'independent_numpy_actor':True,'independent_scalar_stage_checks':True,'recorded_action_native_replay':True,'replacement_policy_rollouts':0,'case_reports':reports};put(e/'RETAINED_RECORD_AUDIT.json',audit)
 selection=frontier.select(seed_rows,source_sha256=dr.sha(e/'development_selection_rows.json'),study_domain='warehouse');put(c/'DEVELOPMENT_SELECTION.json',selection)
 final={'schema':'PCI-FAIR-COHORT-COMPLETE-v1','status':'COMPLETE_AUDITED_COHORT','stage':'development','study_domain':'warehouse','all_owned_worker_processes_closed':True,'trained_policies':144,'training_joint_steps':37748736,'execution_lock_sha256':dr.sha(c/'EXECUTION_LOCK.json'),'source_map_sha256':dr.digest(sources),'case_manifest_sha256':comparisons['case_manifest_sha256'],'absolute_deadline_unix':1000,'completed_unix':999,'analysis_sha256':dr.sha(e/'ANALYSIS.json'),'retained_record_audit_sha256':dr.sha(e/'RETAINED_RECORD_AUDIT.json'),'evaluated_episodes':5760,'evaluation_joint_steps':1474560,'evaluation_agent_decisions':5898240};put(c/'STUDY_COMPLETE.json',final)

def refuse(label,fn):
 try:fn()
 except (ValueError,FileNotFoundError) as e:return {'control':label,'status':'PASS_REFUSED','error':str(e)}
 raise AssertionError('Invalid input accepted: '+label)

def main():
 report={'schema':'PCI-DESCRIPTIVE-REPORT-SYNTHETIC-CONTROLS-v1','synthetic_only':True,'scientific_results_generated':False,'controls':[]}
 with tempfile.TemporaryDirectory(prefix='pci_report_test_') as tmp:
  base=Path(tmp);bundle=base/'bundle';bundle.mkdir();make_fixture(bundle)
  report['controls'].append(refuse('synthetic research input',lambda:dr.validate_bundle(bundle,SOURCE)))
  data=dr.validate_bundle(bundle,SOURCE,_test_only_synthetic=True)
  assert len(data['seed_rows'])==144 and len(data['profile_rows'])==576
  report['controls'].append({'control':'all 5760 synthetic case summaries independently aggregated','status':'PASS'})
  def mutate(path,change,label):
   original=path.read_bytes()
   try:put(path,change(json.loads(original)));report['controls'].append(refuse(label,lambda:dr.validate_bundle(bundle,SOURCE,_test_only_synthetic=True)))
   finally:path.write_bytes(original)
  p=bundle/'cohort/STUDY_COMPLETE.json';mutate(p,lambda v:dict(v,status='INCOMPLETE_REQUIRES_REVIEW'),'incomplete study')
  mutate(p,lambda v:dict(v,schema='WRONG_PRODUCER_SCHEMA'),'wrong producer schema')
  p=bundle/'cohort/evaluation/RETAINED_RECORD_AUDIT.json';original=p.read_bytes();p.unlink();report['controls'].append(refuse('missing retained replay audit',lambda:dr.validate_bundle(bundle,SOURCE,_test_only_synthetic=True)));p.write_bytes(original)
  p=bundle/'cohort/evaluation/development_selection_rows.json';mutate(p,lambda v:v[:-1],'missing seed-candidate row');mutate(p,lambda v:v+[v[0]],'duplicated seed-candidate row')
  p=next((bundle/'cohort/evaluation/cases').glob('*.json'));mutate(p,lambda v:dict(v,totals=dict(v['totals'],deliveries=v['totals']['deliveries']+1)),'tampered case count')
  def rebind_case(path,change,label):
   original=path.read_bytes();cp=bundle/'cohort/evaluation/EVALUATION_COMPLETE.json';completion=cp.read_bytes()
   try:
    v=change(json.loads(original));put(path,v);record=json.loads(completion);record['case_receipts'][v['case']['case_id']]=dr.sha(path);put(cp,record)
    report['controls'].append(refuse(label,lambda:dr.validate_bundle(bundle,SOURCE,_test_only_synthetic=True)))
   finally:path.write_bytes(original);cp.write_bytes(completion)
  shield=next((bundle/'cohort/evaluation/cases').glob('*shieldaware*.json'))
  rebind_case(shield,lambda v:dict(v,totals=dict(v['totals'],physical_replacements=1,total_replacements=1)),'impossible shield correction despite rebound case hash')
  fixed=next((bundle/'cohort/evaluation/cases').glob('*fixed*.json'))
  rebind_case(fixed,lambda v:dict(v,totals=dict(v['totals'],total_replacements=1025)),'replacement count above decision exposure')
  rebind_case(fixed,lambda v:dict(v,totals=dict(v['totals'],total_replacements=100)),'replacement Hamming triangle violation')
  rebind_case(fixed,lambda v:dict(v,schema='WRONG_CASE_SCHEMA'),'wrong case producer schema despite rebound hash')
  p=bundle/'cohort/evaluation/EVALUATION_COMPLETE.json';mutate(p,lambda v:dict(v,trace_intent_receipts=dict(v['trace_intent_receipts'],extra='x')),'extra trace-intent receipt')
  p=bundle/'cohort/DEVELOPMENT_SELECTION.json';mutate(p,lambda v:dict(v,selected_candidate_ids=list(reversed(v['selected_candidate_ids']))),'tampered declared selection')
  path=base/'input_snapshot_control.json';put(path,{'value':1});snapshot=dr.InputSnapshot();snapshot.read(path);put(path,{'value':2})
  report['controls'].append(refuse('input changed after held-body read',snapshot.check))
  _,_,frontier=dr.load_protocol(SOURCE)
  # Independent area: .3*(.02-.01) + .6*(.05-.02), divided by .05 = .42.
  assert abs(frontier.hypervolume([(.01,.3),(.02,.6)],.05)-.42)<1e-14
  assert abs(frontier.hypervolume([(.01,.3),(.02,.2),(.06,.99)],.05)-.24)<1e-14
  assert frontier.hypervolume([], .05)==0
  # Split floating-point area integration need not equal the dominating
  # singleton bit-for-bit; the reporting label uses analytic geometry.
  assert dr.two_point_redundant([(1/40960,53/40960),(3/40960,29/40960)])
  assert dr.two_point_redundant([(.01,.2),(.05,.9)])
  assert dr.two_point_redundant([(.01,0),(.02,.2)])
  assert not dr.two_point_redundant([(.01,.2),(.02,.4)])
  report['controls'].append({'control':'analytic two-point redundancy handles split-rounding, boundary and zero-utility cases','status':'PASS'})
  # Seedwise max/area does not commute with averaging candidate points.
  assert abs(frontier.hypervolume([(0,.2),(.1,1)],.05)-.2)<1e-14
  assert frontier.hypervolume([(.05,.6),(.05,.6)],.05)==0
  report['controls'].append({'control':'selection HV of means differs from mean seedwise HV; selector remains fixed','status':'PASS'})
  report['controls'].append({'control':'independent rectangular-area hypervolume arithmetic; dominated and out-of-box points','status':'PASS'})
  # A separately labeled rendering fixture exercises heterogeneous endpoints.
  # For A/B, seedwise HV is 1/5, 1/5, 2/5; mean-point HV is 8/45.
  heterogeneous=copy.deepcopy(data);fixed_ids=sorted(c['candidate_id'] for c in data['candidates'] if c['family']=='fixed');a,b=fixed_ids[:2]
  patterns={a:[(0,.2),(.1,1),(0,.4)],b:[(.1,1),(0,.2),(.1,.9)]}
  for r in heterogeneous['seed_rows']:
   if r['family']!='fixed':continue
   cost,utility=patterns.get(r['candidate_id'],[(0,0)]*3)[[61001,61002,61003].index(r['training_seed'])]
   nsteps=4*r['joint_steps'];r.update(institutional_replacements=round(cost*nsteps),deliveries=round(utility*nsteps),pickups=round(utility*nsteps),physical_replacements=0,total_replacements=round(cost*nsteps),normalized_cost=cost,normalized_utility=utility,institutional_per_1000_agent_decisions=cost*1000,deliveries_per_1000_joint_steps=utility*4000)
   for profile_row in heterogeneous['profile_rows']:
    if (profile_row['candidate_id'],profile_row['training_seed'])==(r['candidate_id'],r['training_seed']):
     for key in dr.COUNTS:profile_row[key]=r[key]//4
  heterogeneous['selection']=frontier.select(heterogeneous['seed_rows'],source_sha256=dr.digest(heterogeneous['seed_rows']),study_domain='warehouse')
  assert heterogeneous['selection']['families']['fixed']['selected']==[a,b]
  dr.write_report(heterogeneous,base/'synthetic_render',SOURCE)
  with (base/'synthetic_render/frozen_selected_pair_hypervolumes.csv').open() as handle:
   endpoint_rows=[r for r in csv.DictReader(handle) if r['family']=='fixed']
  assert [int(r['training_seed']) for r in endpoint_rows]==[61001,61002,61003]
  for r,expected in zip(endpoint_rows,[1/5,1/5,2/5]):
   assert r['candidate_1']==a and r['candidate_2']==b
   assert abs(float(r['selected_pair_seed_hypervolume'])-expected)<1e-14
   assert abs(float(r['selection_hypervolume_of_mean_points'])-8/45)<1e-14
   assert abs(float(r['mean_of_selected_pair_seed_hypervolumes'])-4/15)<1e-14
   assert r['mean_pair_has_redundant_or_tied_point']=='True'
  report['controls'].append({'control':'heterogeneous frozen-pair CSV endpoints match independent 1/5,1/5,2/5 and 8/45 arithmetic; no reselection','status':'PASS'})
  assert len(list((base/'synthetic_render').glob('*.pdf')))==4 and len(list((base/'synthetic_render').glob('*.svg')))==4
  report['controls'].append({'control':'all vector plot paths execute on visibly labeled synthetic fixtures','status':'PASS','fixture_figures_retained':False})
 report['status']='PASS_SYNTHETIC_REPORTING_CONTROLS'
 report['total_checks']=len(report['controls'])
 report['source_map_sha256']=dr.PINNED_SOURCE
 report['reporter_sha256']=hashlib.sha256((ROOT/'development_report.py').read_bytes()).hexdigest()
 report['test_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (ROOT/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
