"""Artificial software controls only; never reads live research outputs."""
import copy,csv,hashlib,io,json,re,sys,tempfile
from pathlib import Path
import training_diagnostics as td
b=td.b
ROOT=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent/'pci_development_readiness_20261106/pinned_source'

def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def csvbytes(rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return f.getvalue().encode()

def fixture(root):
    sources,study,_=b.load_protocol(SOURCE);defaults=td.runtime_defaults(SOURCE);i=root/'inputs';c=root/'cohort';i.mkdir(parents=True);c.mkdir()
    put(root/'SYNTHETIC_TEST_ONLY.json',{'label':'ARTIFICIAL SOFTWARE CONTROLS, NOT RESEARCH DATA'})
    protocol=study.protocol();put(i/'cases/PROTOCOL_DRAFT.json',protocol);cases=[];rawconfigs={}
    for seed in study.DEVELOPMENT_SEEDS:
        for candidate in study.candidates():
            raw=study.train_config(candidate,seed,study.DEVELOPMENT_SEEDS.index(seed),role='development',steps=262144,lock_hash=b.digest(protocol),domain='warehouse');run=raw['run_id'];path=i/'cases'/(run+'.json');put(path,raw);rawconfigs[run]=raw
            cases.append({'case_id':run,'candidate_id':candidate['candidate_id'],'family':candidate['family'],'training_seed':seed,'domain':'warehouse','n_agents':4,'run_role':'development','total_steps':262144,'path':path.name,'sha256':b.sha(path)})
    ledger={'schema':'PCI-COMPLETE-CASE-MANIFEST-v1','stage':'development','case_count':144,'protocol_canonical_sha256':b.digest(protocol),'cases':cases};put(i/'cases/CASE_MANIFEST.json',ledger)
    cfg={'schema':'PCI-FAIR-COHORT-CONTROLLER-CONFIG-v1','run_id':'synthetic','role':'scientific','stage':'development','study_domain':'warehouse','device':'cuda','out':'/synthetic/cohort','deadline_unix':1000,'source_design_lock':'/synthetic/inputs/SOURCE_DESIGN_LOCK.json','launch_lock':'/synthetic/inputs/LAUNCH_LOCK.json','readiness_receipt':'/synthetic/inputs/CUDA_READINESS.json','protocol_file':'/synthetic/inputs/cases/PROTOCOL_DRAFT.json'};put(i/'CONTROLLER_CONFIG.json',cfg)
    design={'schema':'PCI-COMPLETE-SOURCE-DESIGN-LOCK-v1','source_sha256':sources,'source_map_sha256':b.digest(sources),'protocol_canonical_sha256':b.digest(protocol),'scientific_launch_authorized':True,'engineering_complete':True};put(i/'SOURCE_DESIGN_LOCK.json',design)
    ready={'schema':'PCI-FAIR-CUDA-READINESS-v1','status':'PASS','all_required_methods_pass':True,'source_map_sha256':b.digest(sources),'gpu_uuid':'SYNTHETIC','host_boot_id':'SYNTHETIC'};put(i/'CUDA_READINESS.json',ready)
    bind={'controller_config_sha256':b.digest(cfg),'case_manifest_sha256':b.sha(i/'cases/CASE_MANIFEST.json'),'source_design_lock_sha256':b.sha(i/'SOURCE_DESIGN_LOCK.json'),'readiness_receipt_sha256':b.sha(i/'CUDA_READINESS.json'),'source_map_sha256':b.digest(sources),'out':cfg['out'],'deadline_unix':1000}
    launch={'schema':'PCI-FAIR-SCIENTIFIC-LAUNCH-v1','scientific_launch_authorized':True,'gpu_uuid':'SYNTHETIC','host_boot_id':'SYNTHETIC',**bind};put(i/'LAUNCH_LOCK.json',launch)
    external={cfg[k]:b.sha(i/name) for k,name in [('source_design_lock','SOURCE_DESIGN_LOCK.json'),('launch_lock','LAUNCH_LOCK.json'),('readiness_receipt','CUDA_READINESS.json'),('protocol_file','cases/PROTOCOL_DRAFT.json')]}
    put(c/'EXECUTION_LOCK.json',{'controller_configuration':cfg,'controller_configuration_file_sha256':b.sha(i/'CONTROLLER_CONFIG.json'),'source_sha256':sources,'case_manifest_sha256':bind['case_manifest_sha256'],'authority':{**bind,'external_record_sha256':external}})
    ts={'pci_risk_train/'+name:sources['runtime/training/pci_risk_train/'+name] for name in td.TRAIN_SOURCES};state={'schema':'PCI-FAIR-COHORT-STATE-v1','run_id':'synthetic','status':'EVALUATING','deadline_unix':1000,'host_boot_id':'SYNTHETIC','completed_training':{}}
    for run,raw in rawconfigs.items():
        rootcase=c/'training'/run;case=td.expanded_config(raw,defaults);put(rootcase/'config.json',case)
        system={'versions':{'fixture':'SYNTHETIC'},'dependency_source_sha256':{}};sig=b.digest({'configuration':case,'sources':ts,**system})
        put(rootcase/'provenance.json',{'signature':sig,'sources':ts,'system':system,'version':'PCI-FAIR-INSTITUTIONAL-PPO-ENGINEERING-v1'})
        active=case['method']=='shieldaware' or (case['method']=='direct' and case['auxiliary_coef']>0);rows=[];dual=case['dual_initial']
        for update in range(1,257):
            cost=0. if case['method']=='shieldaware' else .005;used=dual if case['method']=='lagrangian' else 0.
            if case['method']=='lagrangian':dual=min(20.,max(0.,dual+case['dual_lr']*(cost-case['dual_target'])))
            pre=640000 if active else 0;hit=80000 if active else 0;cert=pre//2 if case['method']=='direct' else 0;aux=case['auxiliary_coef']*.1;ppo=1.1 if case['method']=='lagrangian' else 1.
            row={'update':update,'joint_steps':update*1024,'agent_decisions':update*4096,'method':case['method'],'mean_task_reward':.1,'mean_institutional_cost':cost,'mean_product_probe_cost':.01,'sampled_penalty_used':case['penalty_coef'],'dual_used':used,'dual_next':dual,'actual_auxiliary_coefficient':case['auxiliary_coef'],'auxiliary_reference_n4':case['auxiliary_reference_n4'],'auxiliary_reference_ratio':1. if case['auxiliary_reference_n4'] is not None else None,'reward_advantage_sd':.3,'optimizer_steps':1,'mean_approx_kl':.01,'actor_change_l2':.01,'parameter_change_l2':.02,'optimization_requested_rows':pre+hit,'optimization_evaluated_rows':pre-cert,'optimization_cache_hit_rows':hit,'optimization_precompute_rows':pre,'optimization_certified_zero_rows':cert,'elapsed_s':float(update)}
            norms={'reward_policy':1.,'cost_policy':.2 if case['method']=='lagrangian' else 0.,'auxiliary':aux,'reward_value':.1,'cost_value':.1,'entropy':case['entropy_coef'],'ppo_policy':ppo,'preclip':1.,'clipped':.5}
            row.update({'mean_'+k+'_grad_norm':v for k,v in norms.items()});row['mean_auxiliary_to_ppo_grad_norm']=aux/ppo;row['mean_auxiliary_ppo_grad_cosine']=0. if aux else None;rows.append(row)
        (rootcase/'updates.csv').write_bytes(csvbytes(rows))
        files={}
        for name in ('config.json','provenance.json','updates.csv'):files[name]={'bytes':(rootcase/name).stat().st_size,'sha256':b.sha(rootcase/name)}
        for update in range(1,257):
            for folder,suffix in [('rollouts','.npz'),('optimization','.json')]:files[f'{folder}/update_{update:06d}{suffix}']={'bytes':1,'sha256':'f'*64}
        for name in ('final_export.npz','checkpoints/update_000256.pt'):files[name]={'bytes':1,'sha256':'f'*64}
        put(rootcase/'CLOSED_FILES.json',{'schema':'PCI-FAIR-TRAIN-CLOSED-FILES-v1','signature':sig,'files':files,'optional_publication_aliases':{}})
        counters={k:0 for k in td.COUNTERS};counters.update(optimizer_steps=256,proposed_agent_actions=1048576,native_oracle_matches=262144,deployment_shield_uses=262144 if case['method']=='shieldaware' else 0,collection_oracle_requested_rows=262144*(627 if case['method']=='shieldaware' else 2),collection_oracle_callback_calls=32768*(3 if case['method']=='shieldaware' else 2),collection_certified_zero_rows=131072,optimization_oracle_requested_rows=sum(r['optimization_requested_rows'] for r in rows),optimization_oracle_evaluated_rows=sum(r['optimization_evaluated_rows'] for r in rows),optimization_cache_hit_rows=sum(r['optimization_cache_hit_rows'] for r in rows),optimization_precompute_rows=sum(r['optimization_precompute_rows'] for r in rows),optimization_certified_zero_rows=sum(r['optimization_certified_zero_rows'] for r in rows),optimization_oracle_callback_calls=256*33 if active else 0,risk_requested_joint_queries=256*80000 if case['method']=='direct' and active else 0,risk_callback_calls=256 if case['method']=='direct' and active else 0)
        counters['collection_oracle_evaluated_rows']=counters['collection_oracle_requested_rows']-counters['collection_certified_zero_rows']
        done={'schema':'PCI-FAIR-TRAIN-COMPLETE-v1','status':'COMPLETE','version':'PCI-FAIR-INSTITUTIONAL-PPO-ENGINEERING-v1','signature':sig,**{k:case[k] for k in ('run_id','method','domain','seed','run_role','protocol_sha256')},'sources':ts,'joint_steps':262144,'agent_decisions':1048576,'updates':256,'complete_episodes':1024,'final_dual':dual,'counters':counters,'closed_manifest_sha256':b.sha(rootcase/'CLOSED_FILES.json'),'portable_export_sha256':'f'*64,'checkpoint':'checkpoints/update_000256.pt','checkpoint_sha256':'f'*64,'phase_counts':{p:[1024]*64 for p in case['profiles']}}
        put(rootcase/'TRAIN_COMPLETE.json',done);state['completed_training'][run]={'train_complete_sha256':b.sha(rootcase/'TRAIN_COMPLETE.json'),'closed_manifest_sha256':done['closed_manifest_sha256'],'signature':sig,'joint_steps':262144}
    put(c/'COHORT_STATE.json',state)

def refuse(fn,label,expected=None):
    try:fn()
    except (ValueError,KeyError,FileNotFoundError) as err:
        if expected:assert expected in str(err),(label,str(err))
        return {'control':label,'status':'PASS_REFUSED','error':str(err)}
    raise AssertionError('Unexpected acceptance: '+label)

def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='pci_training_diag_control_') as temp:
        root=Path(temp);bundle=root/'bundle';fixture(bundle)
        results.append(refuse(lambda:td.validate(bundle,SOURCE),'Synthetic input refused by public path','SYNTHETIC_INPUT_REFUSED'))
        data=td.validate(bundle,SOURCE,_synthetic=True);assert len(data['runs'])==144 and len(data['trajectories'])==36864;results.append({'control':'144 complete artificial cases and 36864 updates qualify','status':'PASS'})
        case=next(p for p in (bundle/'cohort/training').iterdir() if 'lagrangian_00' in p.name);cfg=b.load(case/'config.json');done=b.load(case/'TRAIN_COMPLETE.json');rows=td.csv_rows(case/'updates.csv')
        for label,change,error in [
            ('Missing update row',lambda rr:rr.pop(),'UPDATE_COUNT'),
            ('Duplicate update coordinate',lambda rr:rr[1].update(update=1),'UPDATE_SEQUENCE'),
            ('Broken projected dual recurrence',lambda rr:rr[0].update(dual_next=5.),'PROJECTED_DUAL_RECURRENCE'),
            ('Clipped norm exceeds declared bound',lambda rr:rr[0].update(mean_clipped_grad_norm=.7),'CLIPPING_BOUND'),
            ('Mean ratio improperly set to missing',lambda rr:rr[0].update(mean_auxiliary_to_ppo_grad_norm=None),'RATIO_MISSINGNESS'),
            ('Impossible extra optimization work',lambda rr:rr[0].update(optimization_requested_rows=1),'OPTIMIZATION_WORK_CONSERVATION')]:
            altered=copy.deepcopy(rows);change(altered);results.append(refuse(lambda:td.validate_updates(altered,cfg,done),label,error))
        altered=copy.deepcopy(rows);altered[0].update(mean_ppo_policy_grad_norm=0.,mean_auxiliary_to_ppo_grad_norm=None,mean_auxiliary_ppo_grad_cosine=0.)
        results.append(refuse(lambda:td.validate_updates(altered,cfg,done),'Undefined zero-PPO cosine refused','ZERO_PPO_COSINE_UNDEFINED'))
        altered_done=copy.deepcopy(done);altered_done['counters']['collection_certified_zero_rows']=262145;altered_done['counters']['collection_oracle_evaluated_rows']=altered_done['counters']['collection_oracle_requested_rows']-262145
        results.append(refuse(lambda:td.validate_updates(rows,cfg,altered_done),'Too many zero-certified probe rows refused','COLLECTION_CERTIFIED_ZERO_BOUND'))
        # Float32 component norms need a relative tolerance when magnitudes grow.
        altered=copy.deepcopy(rows);altered[0].update(mean_reward_policy_grad_norm=100.,mean_cost_policy_grad_norm=1.,mean_ppo_policy_grad_norm=101.00001)
        td.validate_updates(altered,cfg,done);results.append({'control':'Relative tolerance accepts source-compatible float32 norm rounding','status':'PASS'})
        shield=next(p for p in (bundle/'cohort/training').iterdir() if 'shieldaware_00' in p.name);scfg=b.load(shield/'config.json');sdone=b.load(shield/'TRAIN_COMPLETE.json');sr=td.csv_rows(shield/'updates.csv');sr[0]['mean_institutional_cost']=.001
        results.append(refuse(lambda:td.validate_updates(sr,scfg,sdone),'Shield training correction refused','SHIELD_TRAINING_COST_ZERO'))
        path=case/'updates.csv';original=path.read_bytes();path.write_bytes(original.replace(b'0.005,',b'0.006,',1));results.append(refuse(lambda:td.validate(bundle,SOURCE,_synthetic=True),'Tampered diagnostic body fails closure hash','DIAGNOSTIC_BODY_HASH'));path.write_bytes(original)
        statepath=bundle/'cohort/COHORT_STATE.json';original=statepath.read_bytes();state=json.loads(original);state['completed_training'].pop(next(iter(state['completed_training'])));put(statepath,state);results.append(refuse(lambda:td.validate(bundle,SOURCE,_synthetic=True),'Incomplete training inventory','ALL_144_TRAINING_CASES_CLOSED'));statepath.write_bytes(original)
        path=root/'held.json';put(path,{'x':1});snap=b.InputSnapshot();snap.read(path);put(path,{'x':2});results.append(refuse(snap.check,'Changed held-body input refused','INPUT_CHANGED_AFTER_READ'))
        # Two minibatches can have means(aux)=2, means(PPO)=2 but mean ratios=5/3.
        # The recorded estimator must survive unchanged rather than become1.
        direct=next(r for r in data['trajectories'] if r['family']=='direct' and r['lambda']>0);direct['mean_auxiliary_grad_norm']=2.;direct['mean_ppo_policy_grad_norm']=2.;direct['mean_auxiliary_to_ppo_grad_norm']=5/3
        # A zero PPO denominator remains empty, never silently becomes a zero ratio.
        zero=next(r for r in data['trajectories'] if r['family']=='fixed');zero['mean_ppo_policy_grad_norm']=0.;zero['mean_reward_policy_grad_norm']=0.;zero['mean_auxiliary_to_ppo_grad_norm']=None
        td.report(data,root/'render')
        with (root/'render/training_update_diagnostics.csv').open() as handle:export=list(csv.DictReader(handle))
        picked=next(r for r in export if r['run_id']==direct['run_id'] and r['update']==str(direct['update']));assert abs(float(picked['mean_auxiliary_to_ppo_grad_norm'])-5/3)<1e-14
        picked=next(r for r in export if r['run_id']==zero['run_id'] and r['update']==str(zero['update']));assert picked['mean_auxiliary_to_ppo_grad_norm']==''
        results.append({'control':'CSV preserves mean minibatch ratio and missing values; does not use ratio of means','status':'PASS'})
        assert len(list((root/'render').glob('*.pdf')))==6 and len(list((root/'render').glob('*.svg')))==9
        assert len(re.findall(rb'/Type /Page\b',(root/'render/all_candidate_component_gradient_norms.pdf').read_bytes()))==4
        results.append({'control':'Six PDF outputs, nine SVGs and four family pages render on labeled synthetic data; figures removed','status':'PASS'})
    receipt={'schema':'PCI-TRAINING-DIAGNOSTIC-SYNTHETIC-CONTROLS-v1','status':'PASS_SYNTHETIC_CONTROLS','synthetic_only':True,'scientific_results_generated':False,'total_checks':len(results),'controls':results,'source_map_sha256':b.PINNED_SOURCE,'implementation_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('bound_inputs.py','training_diagnostics.py','test_training_diagnostics.py')}}
    (ROOT/'TEST_RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
