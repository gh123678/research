import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(r'C:/Users/Admin/Desktop/research/icrl_softmax')
CODE=ROOT/'results/FP-SURVIVOR-REVIEW-001/codex_worktree/icrl_softmax'
sys.path.insert(0,str(CODE))
from evaluate_fp_xfam_001 import FAMILIES,build_family_task
OUT=ROOT/'results/FP-SURVIVOR-REVIEW-001/codex'
used=[]
def read(p):
    p=ROOT/p; used.append(p); return json.loads(p.read_text())
def value(mdp,pi):
    p=np.asarray(mdp['P']); r=np.asarray(mdp['R']); g=0.7
    rr=(p*r).sum(2); pp=np.einsum('sa,sat->st',pi,p)
    v=np.linalg.solve(np.eye(len(pi))-g*pp,(pi*rr).sum(1))
    return v,rr+g*np.einsum('sat,t->sa',p,v)
def decision(pi,q,e):
    p=pi.copy(); chosen=[]
    for s in range(len(pi)):
        ch=-1
        for j,eta in enumerate((1,.5,.2,.1,.05,.02,.01)):
            z=np.log(pi[s])+eta*q[s]; a=np.exp(z-z.max()); a/=a.sum()
            d=a-pi[s]; lb=float(d@q[s]-e*np.abs(d).sum())
            if lb>0: p[s]=a;ch=j;break
        chosen.append(ch)
    return p,chosen
b=read('results/FP-COST-001/claude/all/task_results.json')
x=read('results/FP-XRULE-002/claude/f1f2_K12/task_results.json')
n=read('results/FP-NETX-002/claude/f1f2_K4/task_results.json')
r=read('results/FP-RERUN-001/claude/fresh_K16/task_results.json')
res={'cost':{},'net_frozen':{},'rerun':{}}
refmap={(a['family'],a['mixing'],a['task_index'],name):rt for a in x['records'] for name,rt in a['routes'].items()}
for fam in ('f1','f2'):
    refs=[rt['perstate|frozen'] for a in x['records'] if a['family']==fam for rt in a['routes'].values()]
    cost=sum(sum(e['items_this_step'] for e in rt['steps']) for rt in refs)
    refgain=np.mean([sum(e.get('total_value_gain',0) for e in rt['steps']) for rt in refs])
    ar=[]
    for run in b['runs']:
        if run['family']!=fam:continue
        rr=[rt['conj|frozen'] for a in run['records'] for rt in a['routes'].values()]
        ss=[e for rt in rr for e in rt['steps']]
        d={'chains':run['chains'],'items':sum(e['items_this_step'] for e in ss),'gain':float(np.mean([sum(e.get('total_value_gain',0) for e in rt['steps']) for rt in rr])),'emitted':sum(e['emitted'] for e in ss),'steps':len(ss)}
        if run['chains']==16384:
            d['full_record_matches']=sum(rt['conj|frozen']==refmap[(fam,a['mixing'],a['task_index'],name)]['conj|frozen'] for a in run['records'] for name,rt in a['routes'].items())
        ar.append(d)
    top=max(b['runs'],key=lambda z:z['chains'] if z['family']==fam else -1)
    diffs=[]
    for a in top['records']:
        if a['family']!=fam:continue
        # group both routes within an environment, not independent samples
        diffs.append(np.mean([refmap[(fam,a['mixing'],a['task_index'],name)]['perstate|frozen']['total_value_gain']-rt['conj|frozen']['total_value_gain'] for name,rt in a['routes'].items()]))
    res['cost'][fam]={'perstate_items':cost,'perstate_gain':float(refgain),'rungs':ar,'top_cost_ratio':ar[-1]['items']/cost,'top_paired_environment_mean_gain_diff':float(np.mean(diffs)),'top_paired_environment_sd':float(np.std(diffs,ddof=1)),'environments':len(diffs),'top_env_wins':sum(v>0 for v in diffs),'top_env_losses':sum(v<0 for v in diffs)}
for fam in ('f1','f2'):
    ctr={'paired_routes':0,'paired_step_decisions':0,'emission_disagreements':0,'row_eta_disagreements':0,'record_emission_disagreements':0,'replay_decision_mismatches':0,'steps_checked':0,'max_value_delta_error':0.,'min_value_delta':float('inf'),'coverage_failures_from_solve':0,'max_q_gap_own_policy':0.,'emitted_numpy':0,'emitted_network':0}
    for a in n['records']:
        if a['family']!=fam:continue
        mdp,pi0,_=build_family_task(FAMILIES[fam],a['mixing'],a['task_index'])
        for route,rt in a['routes'].items():
            ctr['paired_routes']+=1
            recs=[]
            for producer in ('numpy','network'):
                pi=pi0.copy(); previous,_=value(mdp,pi); ds=[]
                es=rt[producer+'|frozen']['steps']
                ctr['emitted_'+producer]+=sum(e['emitted'] for e in es)
                for e in es:
                    q=np.asarray(e['q_hat']); eq=e['e_q']; _,qt=value(mdp,pi)
                    ctr['coverage_failures_from_solve']+=int(np.max(np.abs(q-qt))>eq)
                    nxt,choices=decision(pi,q,eq); emit=any(j>=0 for j in choices)
                    ctr['replay_decision_mismatches']+=int(emit!=e['emitted'] or sum(j>=0 for j in choices)!=e['states_updated'])
                    now,_=value(mdp,nxt); dv=now-previous
                    if emit:
                        ctr['max_value_delta_error']=max(ctr['max_value_delta_error'],float(np.max(np.abs(dv-np.asarray(e['value_delta'])))))
                        ctr['min_value_delta']=min(ctr['min_value_delta'],float(dv.min()))
                    ds.append((emit,choices)); pi=nxt;previous=now;ctr['steps_checked']+=1
                recs.append(ds)
            ctr['record_emission_disagreements']+=int(rt['numpy|frozen']['emitted_steps']!=rt['network|frozen']['emitted_steps'])
            for j,(a1,a2) in enumerate(zip(*recs)):
                ctr['paired_step_decisions']+=1;ctr['emission_disagreements']+=int(a1[0]!=a2[0]);ctr['row_eta_disagreements']+=int(a1[1]!=a2[1])
                q1=np.asarray(rt['numpy|frozen']['steps'][j]['q_hat']);q2=np.asarray(rt['network|frozen']['steps'][j]['q_hat'])
                ctr['max_q_gap_own_policy']=max(ctr['max_q_gap_own_policy'],float(np.max(np.abs(q1-q2))))
    res['net_frozen'][fam]=ctr
for cell in ('numpy|frozen','numpy|L12'):
    rr=[rt[cell] for a in r['records'] for rt in a['routes'].values()]
    res['rerun'][cell]={'emitted':sum(sum(e['emitted'] for e in rt['steps']) for rt in rr),'gain':float(np.mean([sum(e.get('total_value_gain',0) for e in rt['steps']) for rt in rr]))}
res['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used}
(OUT/'independent_checks.json').write_text(json.dumps(res,indent=2,default=lambda o:o.item()),encoding='utf-8')
print(json.dumps(res,indent=2,default=lambda o:o.item()))

