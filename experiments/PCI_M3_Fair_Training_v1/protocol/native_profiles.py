#!/usr/bin/env python3
"""Record root-selected qualified native profiles, never infer scientific approval."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build():
    receipt=ROOT/'validation/environments/REFERENCE_PROFILE_QUALIFICATION.json'
    plan=ROOT/'validation/environments/REFERENCE_PROFILE_QUALIFICATION_PLAN.json'
    data=json.loads(receipt.read_text())
    if data.get('schema')!='PCI-RWARE-REFERENCE-QUALIFICATION-v1' or data.get('status')!='COMPLETE_REFERENCE_SCREEN_NOT_SCIENTIFIC_APPROVAL':
        raise ValueError('COMPLETE_REFERENCE_SCREEN_REQUIRED')
    if sha(plan)!=data['plan_sha256'] or data['native_reward_only'] is not True or data['no_M3_learner_or_holdout_data_used'] is not True:
        raise ValueError('REFERENCE_SCREEN_IDENTITY')
    for rel,digest in data['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('REFERENCE_ADAPTER_SOURCE_CHANGED')
    result={'schema':'PCI-NATIVE-RWARE-PROFILE-COLLECTION-v1','status':'ROOT_SELECTED_BEFORE_NEW_M3_OUTCOMES',
            'scientific_launch_authorized':False,'source_design_lock_required':True,
            'basis':'Root selected the qualified small-layout transport-band candidate in this revision; this records that scope and does not authorize training.',
            'qualification_sha256':sha(receipt),'qualification_plan_sha256':sha(plan),
            'all_reference_cells_retained':len(data['cells']),'all_reference_episodes_retained':len(data['rows']),
            'profiles_by_n':{}}
    for n in (4,8,12):
        cell=next(c for c in data['cells'] if c['profile_id']=='T_q1_w64_c1' and c['layout']=='small' and c['n_agents']==n)
        if cell['qualification_pass'] is not True or not all(cell['checks'].values()):raise ValueError('SELECTED_CELL_NOT_QUALIFIED')
        pid=f'rware_small_transport_q1_w64_c1_n{n}'
        cfg={'layout':'small','n_agents':n,'horizon':256,'quota_window':64,'quota':1,'capacity':1,
             'rule_change':False,'zone_mode':'transport_bands','reward_mode':'native'}
        result['profiles_by_n'][str(n)]={'schema':'PCI-NATIVE-RWARE-SCIENTIFIC-PROFILES-v1',
            'status':'ROOT_APPROVED_FOR_SOURCE_DESIGN_LOCK','scientific_launch_authorized':False,
            'qualification_sha256':sha(receipt),'qualification_sources':data['sources'],
            'qualification_cell':cell,'profiles':{pid:cfg},'env_profile_ids':[pid]*8,
            'institution_semantics':'Per-agent epoch entry quotas and capacity on simultaneous entries into native transport bands; not standing occupancy.',
            'task_reward':'Exact native RWARE INDIVIDUAL delivery reward; no pickup, distance or potential shaping.',
            'scope':'Only this explicitly qualified small-layout cell; tiny-layout and other failed reference cases remain in the full screen.'}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=build()
    if a.out.exists():raise RuntimeError('Use a new profile manifest path instead of replacing a reviewed selection')
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'agent_counts':list(result['profiles_by_n']),'scientific_launch_authorized':False}))
