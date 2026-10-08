"""Separate controller process, intentionally no training-seed replication."""
import argparse
from .evaluate import evaluate_run,HEURISTICS
from .environment import EnvConfig
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',choices=HEURISTICS,required=True)
    p.add_argument('--layout',required=True);p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--small',action='store_true');a=p.parse_args()
    evaluate_run(a.out,a.episodes,not a.small,heuristic_name=a.controller,base_config=EnvConfig(layout=a.layout))
