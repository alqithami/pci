"""Post-audit follow-up. Never writes to the frozen original experiment."""
import os,sys
from pathlib import Path
BASE=Path(os.environ.get('PCI_BASE','/home/ubuntu/PCI_AAMAS27_Rebuild_v1')).resolve()
if not (BASE/'pci_bench/model.py').is_file():
    raise RuntimeError('Set PCI_BASE to the intact PCI_AAMAS27_Rebuild_v1 directory.')
sys.path.insert(0,str(BASE))
ROOT=Path(__file__).resolve().parents[1]
