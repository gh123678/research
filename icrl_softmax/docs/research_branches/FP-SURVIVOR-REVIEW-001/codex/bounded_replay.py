# Bounded replay of existing frozen records only. No L12S call or new experiment.
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(r'C:/Users/Admin/Desktop/research/icrl_softmax');CODE=ROOT/'results/FP-SURVIVOR-REVIEW-001/codex_worktree/icrl_softmax'
sys.path.insert(0,str(CODE))
from evaluate_fp_xfam_001 import FAMILIES,build_family_task,training_batch,run_route,vectorised_batch_generic
from fp_certfix_first_n import first_visit_batch,step_seed_parts
import fixed_policy_expected_sarsa_scaled as fs
from check_survivors import value,decision
OUT=ROOT/'results/FP-SURVIVOR-REVIEW-001/codex'
def cert(q,pi,b,fam,delta):
    # Independently written first-visit MP frozen arithmetic.
    s,a=pi.shape;flat=b['states']*a+b['actions'];y=b['rewards']+fam['gamma']*(pi*q).sum(1)[b['next_states']]-q.reshape(-1)[flat]
    logs=np.log(4*s*a/delta);ep=[]
    for k in range(s*a):
        v=y[flat==k];assert len(v)>=2000
        ep.append(abs(float(v.mean()))+np.sqrt(2*v.var(ddof=1)*logs/len(v))+(7/3)*20*logs/(len(v)-1))
    return max(ep)/(1-fam['gamma'])
def conj(pi,q,e):
    for eta in (1,.5,.2,.1,.05,.02,.01):
        z=np.log(pi)+eta*q;w=np.exp(z-z.max(1,keepdims=True));p=w/w.sum(1,keepdims=True);dp=p-pi
        if np.all((dp*q).sum(1)-e*np.abs(dp).sum(1)>0):return p,True
    return pi.copy(),False
cost=json.loads((ROOT/'results/FP-COST-001/claude/all/task_results.json').read_text())
net=json.loads((ROOT/'results/FP-NETX-002/claude/f1f2_K4/task_results.json').read_text())
results=[];started=time.time()
# Select the longest existing expected_exact route in each unverified high tier; deterministic tie order.
for run in cost['runs']:
    if run['chains']==16384:continue
    fam=FAMILIES[run['family']];route='expected_exact'
    rec=max(run['records'],key=lambda a:len(a['routes'][route]['conj|frozen']['steps']))
    mdp,pi,rng=build_family_task(fam,rec['mixing'],rec['task_index'])
    from evaluate_fixed_policy_q_routes import policy_quantities
    mu=np.asarray(policy_quantities(mdp,pi)['mu_state']);tr=training_batch(mdp,pi,mu,rng);v,_=value(mdp,pi)
    out={'kind':'cost','family':run['family'],'chains':run['chains'],'mixing':rec['mixing'],'task_index':rec['task_index'],'route':route,'steps':0,'max_eq_error':0.,'max_value_error':0.,'decision_mismatches':0};t=time.time()
    for entry in rec['routes'][route]['conj|frozen']['steps']:
        q=run_route(route,fam,pi,tr);raw=vectorised_batch_generic(mdp,build_family_task(fam,rec['mixing'],rec['task_index'])[1],mu,step_seed_parts(fs.SEED,90417,rec['mixing'],rec['task_index'],entry['step']),run['chains'],fam,64)
        batch,_=first_visit_batch(raw,64,n_states=pi.shape[0],n_actions=pi.shape[1]);eq=cert(q,pi,batch,fam,cost['delta_step']);new,em=conj(pi,q,eq);nv,_=value(mdp,new)
        out['max_eq_error']=max(out['max_eq_error'],abs(eq-entry['e_q']));out['decision_mismatches']+=int(em!=entry['emitted'])
        if em:out['max_value_error']=max(out['max_value_error'],float(np.max(np.abs(nv-v-np.asarray(entry['value_delta'])))))
        pi=new;v=nv;out['steps']+=1;del raw,batch
    out['seconds']=time.time()-t;results.append(out);print(out,flush=True)
    (OUT/'bounded_replay.json').write_text(json.dumps(results,indent=2))
# Re-run both actual producers on first record in each family, frozen only, all its steps/routes.
from evaluate_fp_netxfam_001 import make_networks,network_qhat_generic
for famname in ('f1','f2'):
    fam=FAMILIES[famname];rec=next(r for r in net['records'] if r['family']==famname);mdp,base,rng=build_family_task(fam,rec['mixing'],rec['task_index']);mu=np.asarray(policy_quantities(mdp,base)['mu_state']);tr=training_batch(mdp,base,mu,rng);nets=make_networks(fam)
    out={'kind':'network','family':famname,'mixing':rec['mixing'],'task_index':rec['task_index'],'steps':0,'max_q_error':0.,'max_eq_error':0.,'decision_mismatches':0};t=time.time()
    for route,rt in rec['routes'].items():
        for producer in ('numpy','network'):
            pi=base.copy()
            for entry in rt[producer+'|frozen']['steps']:
                q=run_route(route,fam,pi,tr) if producer=='numpy' else network_qhat_generic(nets[route],pi,tr)
                raw=vectorised_batch_generic(mdp,base,mu,step_seed_parts(fs.SEED,90417,rec['mixing'],rec['task_index'],entry['step']),16384,fam,64);batch,_=first_visit_batch(raw,64,n_states=pi.shape[0],n_actions=pi.shape[1]);eq=cert(q,pi,batch,fam,net['delta_step']);new,choices=decision(pi,q,eq);em=any(j>=0 for j in choices)
                out['max_q_error']=max(out['max_q_error'],float(np.max(np.abs(q-np.asarray(entry['q_hat'])))));out['max_eq_error']=max(out['max_eq_error'],abs(eq-entry['e_q']));out['decision_mismatches']+=int(em!=entry['emitted']);out['steps']+=1;pi=new
                del raw,batch
    out['seconds']=time.time()-t;results.append(out);print(out,flush=True);(OUT/'bounded_replay.json').write_text(json.dumps(results,indent=2))
print('elapsed',time.time()-started,flush=True)
