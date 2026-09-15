"""Independent retrospective replay, no imports from either author's task directory.
Run from the isolated icrl_softmax directory. Shared model and array estimator define
the objects under test; certificate sampler, extraction, decision and audit are new.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True
from evaluate_fp_xfam_001 import FAMILIES, build_family_task, training_batch, run_route
from evaluate_fixed_policy_q_routes import policy_quantities
from model import EndToEndMaskedSoftmaxExpectedSARSA, EndToEndFiniteSoftmaxExpectedSARSA

HOST = Path('C:/Users/Admin/Desktop/research/icrl_softmax')
ETAS = (1., .5, .2, .1, .05, .02, .01)

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ah(*aa):
    h = hashlib.sha256()
    for a in aa:
        a = np.ascontiguousarray(a)
        h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def sample(mdp, beta, mu, seed, chains=16384, length=64):
    """Independent implementation preserving the frozen stream's draw ordering.
    Time-major simulation, then transpose to chain-major stored order.
    """
    rng = np.random.default_rng(seed)
    S,A = beta.shape
    pcdf = np.cumsum(np.asarray(mdp['P'], float), axis=-1)
    acdf = np.cumsum(beta, axis=-1); acdf[:,-1] = 1.
    s = np.empty((length,chains), np.int32)
    a = np.empty_like(s); nxt = np.empty_like(s); rew = np.empty((length,chains))
    current = rng.choice(S, size=chains, p=mu).astype(np.int32)
    for t in range(length):
        u = rng.random(chains)
        action = np.minimum((acdf[current] < u[:,None]).sum(1), A-1)
        v = rng.random(chains)
        following = np.minimum((pcdf[current,action] < v[:,None]).sum(1), S-1).astype(np.int32)
        s[t] = current; a[t] = action; nxt[t] = following
        rew[t] = mdp['R'][current,action,following]
        current = following
    return {k: np.ascontiguousarray(v.T).reshape(-1) for k,v in
            [('states',s),('actions',a),('next_states',nxt),('rewards',rew)]}

def extract(raw, A, d, length=64):
    """Unique (chain,pair) keys select their first source index, then restore order."""
    pair = raw['states'].astype(np.int64)*A+raw['actions']
    keys = (np.arange(pair.size)//length)*d+pair
    _, first = np.unique(keys, return_index=True)
    first.sort()
    kept = {k:v[first] for k,v in raw.items()}
    group = pair[first]
    return kept, [np.flatnonzero(group==x) for x in range(d)]

def cert(q, pi, kept, members, gamma, bound, delta):
    v = np.sum(pi*q,axis=1)
    y = kept['rewards']+gamma*v[kept['next_states']]-q[kept['states'],kept['actions']]
    n = np.array([len(i) for i in members])
    if n.min()<2000: return None
    m = np.array([np.mean(y[i]) for i in members])
    vv = np.array([np.var(y[i],ddof=1) for i in members])
    width = 2*(bound+gamma*np.max(np.abs(v))+np.max(np.abs(q)))
    log = math.log(2*len(n)/delta)
    rad = np.sqrt(2*vv*log/n)+(7./3)*width*log/(n-1)
    eps = np.abs(m)+rad
    return dict(pair_sizes=n,pair_means=m,pair_vars=vv,pair_radii=rad,pair_eps=eps,
                e_q=float(eps.max()/(1-gamma)),y_range=float(width))

def candidates(pi,q):
    for eta in ETAS:
        logits = np.log(pi)+eta*q
        weights = np.exp(logits-np.max(logits,axis=1,keepdims=True))
        yield eta,weights/np.sum(weights,axis=1,keepdims=True)

def decision(pi,q,eq):
    new=pi.copy(); chosen=[None]*len(pi); table=[]
    for eta,c in candidates(pi,q):
        u=c-pi
        lb=np.sum(u*q,axis=1)-eq*np.sum(np.abs(u),axis=1)
        table.append(lb)
        for s in range(len(pi)):
            if chosen[s] is None and lb[s]>0:
                chosen[s]=eta;new[s]=c[s]
    return new,chosen,np.array(table).T

def qv(mdp,pi,gamma):
    p=np.asarray(mdp['P'],float); reward=np.sum(p*np.asarray(mdp['R'],float),axis=2)
    kernel=np.einsum('sa,sat->st',pi,p)
    v=np.linalg.solve(np.eye(len(pi))-gamma*kernel,np.sum(pi*reward,axis=1))
    return reward+gamma*np.einsum('sat,t->sa',p,v),v

def forward(net,pi,tensors):
    q=torch.zeros(pi.shape,dtype=torch.float32)
    pol=torch.as_tensor(pi,dtype=torch.float32)
    with torch.no_grad():
        for _ in range(160):
            q,_=net(q,*tensors,pol)
    return q.numpy().astype(float)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--threads',type=int,default=6)
    ap.add_argument('--bridge',action='store_true');args=ap.parse_args()
    if args.output.exists(): raise RuntimeError('Refuse to overwrite an existing run directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(6)
    paths = [HOST/'results'/task/'claude'/sub for task,subs in [
        ('FP-COMPOSE-001',['formal/task_results.json','formal/input_manifest.json','verification/selfcheck_c1.json']),
        ('FP-COMPOSE-002',['formal/task_results.json','formal/input_manifest.json','verification/dimension_scaling.json'])]
        for sub in subs]
    paths += [HOST/'docs/research_branches/FP-COMPOSE-002/claude/dimension_scaling.md',
              HOST/'results/FP-COMPOSE-001/codex_worktree/icrl_softmax/docs/research_tasks/FP-COMPOSE-001.md',
              PROJECT/'docs/research_tasks/FP-COMPOSE-REVIEW-003.md',Path(__file__)]
    meta=dict(input_sha256={str(p):sha(p) for p in paths},argv=sys.argv,python=sys.version,
              numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),
              threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads())
    dump(args.output/'manifest.json',meta)
    bundle=json.loads(paths[3].read_text(encoding='utf-8'))
    old=json.loads(paths[0].read_text(encoding='utf-8'))
    manifests=json.loads(paths[4].read_text(encoding='utf-8'))['records']
    manifests_old=json.loads(paths[1].read_text(encoding='utf-8'))['records']
    key=lambda r:(r['family'],r['mixing'],r['task_index'])
    oldmap={key(r):r for r in old['records']};mm={key(r):r for r in manifests}
    assert {key(r):r for r in manifests_old}==mm,'001/002 input manifest disagreement'
    fields='q_hat pair_sizes pair_means pair_vars pair_radii pair_eps e_q y_range delta_dir lb_table rows_eta update_mask pi_before pi_after emitted states_updated'.split()
    comparisons=0
    for rec in bundle['records']:
        for rt,cc in rec['routes'].items():
            for name,c in cc.items():
                previous=oldmap[key(rec)]['routes'][rt][name]['steps']
                assert len(previous)==4
                for i,e in enumerate(previous):
                    for f in fields: assert e[f]==c['steps'][i][f],(key(rec),rt,name,i,f)
                    comparisons+=1
    dump(args.output/'cross_bundle.json',dict(steps= comparisons,fields=fields,verdict='PASS'))
    stats=dict(records=0,batches=0,network_forwards=0,producer_steps=0,value_steps=0,
               max_q_diff=0.,max_eq_diff=0.,max_value_diff=0.,max_pi_diff=0.,
               min_safety=1e100,min_value=1e100,baseline_steps=0,full_batch_hashes=[],checks=0)
    start=time.perf_counter();loc=None
    def check(cond,what):
        stats['checks']+=1
        if not bool(cond): raise AssertionError((loc,what))
    def close(a,b,tol,what):
        diff=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        check(math.isfinite(diff) and diff<=tol,(what,diff,tol));return diff
    try:
        for rec in bundle['records'][:args.limit or None]:
            fam=FAMILIES[rec['family']];g=fam['gamma'];S,A=fam['n_states'],fam['n_actions']
            mdp,beta,rng=build_family_task(fam,rec['mixing'],rec['task_index'])
            mu=np.asarray(policy_quantities(mdp,beta)['mu_state'],float)
            train=training_batch(mdp,beta,mu,rng)
            man=mm[key(rec)];loc=key(rec)
            check(ah(mdp['P'],mdp['R'])[:16]==man['mdp_hash'],'mdp_hash')
            check(ah(beta)[:16]==man['policy0_hash'],'policy0_hash')
            check(ah(*(train[k] for k in ['states','actions','rewards','next_states']))[:16]==man['train_hash'],'train_hash')
            nets={'expected_exact':EndToEndMaskedSoftmaxExpectedSARSA(gamma=g,alpha=.65),
                  'expected_finite':EndToEndFiniteSoftmaxExpectedSARSA(gamma=g,alpha=.65,zeta=8,xi=8,tau=8)}
            tensors=[torch.as_tensor(train[k],dtype=torch.float32 if k=='rewards' else torch.long)
                     for k in ['states','actions','rewards','next_states']]
            if args.bridge:
                result={}
                for rt,net in nets.items():
                    outputs={};times={}
                    for threads in [6,1]:
                        torch.set_num_threads(threads);ts=time.perf_counter();outputs[threads]=forward(net,beta,tensors);times[threads]=time.perf_counter()-ts
                    result[rt]=dict(max_thread_difference=float(np.max(np.abs(outputs[6]-outputs[1]))),seconds=times,
                        sealed_difference_6=float(np.max(np.abs(outputs[6]-rec['routes'][rt]['network|perstate|L12']['steps'][0]['q_hat']))))
                dump(args.output/'thread_bridge.json',result);print(json.dumps(result),flush=True);return
            policies={(rt,name):beta.copy() for rt,cc in rec['routes'].items() for name in cc}
            live=set(policies)
            for step in range(1,13):
                expected={(rt,name) for rt,cc in rec['routes'].items() for name,c in cc.items() if len(c['steps'])>=step}
                check(live==expected,'live_cell_set')
                if not live: break
                seed=[20260911,90417,int(round(100*rec['mixing'])),rec['task_index'],step]
                raw=sample(mdp,beta,mu,seed);kept,members=extract(raw,A,S*A)
                stats['batches']+=1
                full=ah(*(raw[k] for k in ['states','actions','rewards','next_states']))
                prefix=ah(raw['states'][:1000],raw['next_states'][:1000])[:16]
                stats['full_batch_hashes'].append(dict(record=list(key(rec)),step=step,sha256=full))
                for rt,name in sorted(live):
                    loc=(*key(rec),rt,name,step);e=rec['routes'][rt][name]['steps'][step-1]
                    pi=policies[(rt,name)];stats['max_pi_diff']=max(stats['max_pi_diff'],close(pi,e['pi_before'],1e-12,'pi_before'))
                    check(prefix==e['batch_hash'],'sealed_batch_prefix_hash')
                    if name.startswith('network'):
                        q=forward(nets[rt],pi,tensors);stats['network_forwards']+=1
                    else: q=run_route(rt,fam,pi,train).reshape(S,A)
                    stats['max_q_diff']=max(stats['max_q_diff'],close(q,e['q_hat'],1e-9,'qhat'))
                    ct=cert(q,pi,kept,members,g,fam['reward_bound'],.0125)
                    check(ct is not None,'support')
                    for f in ['pair_sizes','pair_means','pair_vars','pair_radii','pair_eps','y_range']:
                        close(ct[f],e[f],0 if f=='pair_sizes' else 1e-10,f)
                    stats['max_eq_diff']=max(stats['max_eq_diff'],close(ct['e_q'],e['e_q'],1e-10,'eq'))
                    new,eta,table=decision(pi,q,ct['e_q']);emit=any(x is not None for x in eta)
                    check(eta==e['rows_eta'] and emit==e['emitted'],'decision')
                    check([x is not None for x in eta]==e['update_mask'],'mask')
                    close(table,e['lb_table'],1e-10,'lb_table');close(new,e['pi_after'],1e-12,'pi_after')
                    truth,v=qv(mdp,pi,g);close(truth,e['audit']['q_pi_audit'],1e-10,'truth_q')
                    safety=ct['e_q']-float(np.max(np.abs(q-truth)))
                    stats['min_safety']=min(stats['min_safety'],safety)
                    check(safety>1e-10,'coverage_or_boundary')
                    if emit:
                        _,vnext=qv(mdp,new,g);dv=vnext-v
                        stats['max_value_diff']=max(stats['max_value_diff'],close(dv,e['audit']['value_delta'],1e-10,'value_delta'))
                        stats['min_value']=min(stats['min_value'],float(dv.min()))
                        check(dv.min()>1e-10,'degradation_or_boundary')
                        stats['value_steps']+=1;policies[(rt,name)]=new
                    else: live.remove((rt,name))
                    stats['producer_steps']+=1;stats['baseline_steps']+=int(step<=4)
            stats['records']+=1;stats['wall_seconds']=time.perf_counter()-start
            dump(args.output/'progress.json',stats)
            print(json.dumps({k:stats[k] for k in ['records','batches','producer_steps','network_forwards','wall_seconds','max_q_diff']}),flush=True)
        if not args.limit:
            check([stats[k] for k in ['records','batches','producer_steps','network_forwards','baseline_steps']]==[96,1136,4534,2267,1536],'full_scope_counts')
            check(stats['batches']*1048576==bundle['cost']['items_unique'],'unique_items')
        modules={}
        for module in list(sys.modules.values()):
            f=getattr(module,'__file__',None)
            if f and Path(f).suffix=='.py' and Path(f).is_relative_to(PROJECT):
                pp=Path(f);relative=str(pp.relative_to(PROJECT)).replace('\\','/')
                check(not ('/FP-COMPOSE-001/claude/' in relative or '/FP-COMPOSE-002/claude/' in relative),'forbidden_import')
                modules[relative]=sha(pp)
        dump(args.output/'shared_modules.json',modules)
        stats['verdict']='PASS';stats['limited']=bool(args.limit);dump(args.output/'result.json',stats)
        print('PASS',flush=True)
    except Exception as exc:
        dump(args.output/'failure.json',dict(location=loc,error=repr(exc),stats=stats))
        raise

if __name__=='__main__':main()
