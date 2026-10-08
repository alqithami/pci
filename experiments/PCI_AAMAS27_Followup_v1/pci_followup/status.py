import json
from pathlib import Path
from . import ROOT
r=ROOT/'results/full'
if not (r/'plan.json').exists():
    print('Full plan not started. Inspect validation/launch.log and smoke status.')
else:
    p=json.loads((r/'plan.json').read_text());n=len(p['runs'])
    for tag in ('TRAIN_COMPLETE','EVAL_COMPLETE','FAILED'):
        fs=list((r/'runs').glob(f'*/{tag}.json'));print(tag,len(fs),'/',n)
        if tag=='FAILED':
            for f in fs:print(f.parent.name,f.read_text())
    fs=list((r/'certified').glob('n*/COMPLETE.json'));print('Certified full episodes:',len(fs),'/12')
    for name in ('SUITE_FAILED.json','SUITE_COMPLETE.json','analysis/ANALYSIS_REPORT.json'):
        if (r/name).exists():print(name,(r/name).read_text())
for n in ('validation/STATE.txt','validation/EXIT.txt'):
    if (ROOT/n).exists():print(n,(ROOT/n).read_text())
