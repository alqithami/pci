"""Standard-library provenance test with explicitly stubbed module identities.

This verifies which namespace system_meta reads. It is not an observation of
an installed NumPy/Torch runtime and performs no numerical or GPU computation.
"""
from pathlib import Path
import hashlib, importlib.util, json, platform, sys, types
ROOT=Path(__file__).resolve().parents[1]

def run():
    path=ROOT/'training/pci_risk_train/io.py'
    spec=importlib.util.spec_from_file_location('risk_io_provenance_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_modules={name:sys.modules.get(name) for name in ('numpy','torch')}
    np=types.ModuleType('numpy');np.__version__='1.26.4';np.__file__='/literal/imported/numpy/__init__.py'
    torch=types.ModuleType('torch');torch.__version__='2.7.1';torch.__file__='/literal/imported/torch/__init__.py'
    original_version=module.md.version
    try:
        sys.modules['numpy']=np;sys.modules['torch']=torch
        dist={'numpy':'2.4.1','torch':'2.7.1','scipy':'1.0','matplotlib':'1.0','pytest':'1.0'}
        module.md.version=lambda name:dist[name]
        first=module.system_meta()
        assert first['versions']=={'numpy':'1.26.4','torch':'2.7.1','python':platform.python_version()}
        assert first['distribution_metadata_versions']['numpy']=='2.4.1'
        assert first['imported_modules']['numpy']['file']==np.__file__
        assert first['module_distribution_version_discrepancies']==[{
            'module':'numpy','imported_version':'1.26.4','distribution_metadata_version':'2.4.1','imported_file':np.__file__}]
        dist['numpy']='99.0';second=module.system_meta()
        assert second['versions']==first['versions']
        assert second['distribution_metadata_versions']['numpy']=='99.0'
        np.__version__='1.26.5';third=module.system_meta()
        assert third['versions']['numpy']=='1.26.5' and third['versions']!=first['versions']
        assert first['python_executable']==sys.executable
    finally:
        module.md.version=original_version
        for name,old in old_modules.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old
    return {'schema':'PCI-M2-SYSTEM-METADATA-SOURCE-CONTROL-v1','status':'PASS',
        'io_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':['Actual imported versions are distinguished from conflicting distribution metadata',
                  'Imported module __file__ identities are retained',
                  'Changing only distribution-index metadata leaves signature versions unchanged',
                  'Changing actual imported version changes signature versions',
                  'Running Python version and executable are recorded'],
        'scope':'Source behavior under explicit literal stubs; no claim about installed runtime versions',
        'new_worlds_training_RNG_GPU_or_device_calls':0}

if __name__=='__main__':
    report=run();out=ROOT/'training_controls/SYSTEM_METADATA_SOURCE_CHECK.json'
    if out.exists():raise RuntimeError('New output required; do not overwrite prior provenance test')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
