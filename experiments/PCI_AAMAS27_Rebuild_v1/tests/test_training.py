import json,copy
from pathlib import Path
import numpy as np
import torch
import pytest
from pci_bench.train import train
from pci_bench.io import read_csv_strict

ROOT=Path(__file__).resolve().parents[1]

def test_actual_ppo_updates_and_resumes(tmp_path):
    cfg=json.loads((ROOT/'configs/base.json').read_text())
    cfg.update(total_steps=256,n_envs=2,rollout_steps=32,hidden=16,minibatch_groups=32,update_epochs=2,run_id='unit_ppo')
    cfg['env']['horizon']=64
    result=train(cfg,tmp_path)
    start=torch.load(tmp_path/'policy_0.pt',weights_only=True)
    end=torch.load(tmp_path/'checkpoint.pt',weights_only=False)['model']
    for i in range(3):
        keys=[k for k in start if k.startswith(f'heads.{i}.')]
        assert sum(float((end[k]-start[k]).abs().sum()) for k in keys)>0
    assert result['env_steps']==256
    rows=read_csv_strict(tmp_path/'updates.csv')
    assert len(rows)==4 and all(float(r['parameter_change_l2'])>0 for r in rows)
    repeated=train(cfg,tmp_path)
    assert repeated['checkpoint_sha256']==result['checkpoint_sha256']
    changed=copy.deepcopy(cfg);changed['seed']=1
    with pytest.raises(RuntimeError):train(changed,tmp_path)
