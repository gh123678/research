"""Recompute the author's explicit dimension model; do not validate it by reproducing it."""
import json, math, sys, hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(PROJECT))
from evaluate_fp_xfam_001 import FAMILIES,build_family_task
HOST=Path('C:/Users/Admin/Desktop/research/icrl_softmax')
BASE=HOST/'results/FP-COMPOSE-002/claude'
b=json.loads((BASE/'formal/task_results.json').read_text(encoding='utf-8'))
reference=json.loads((BASE/'verification/dimension_scaling.json').read_text(encoding='utf-8'))
outdir=PROJECT/'results/FP-COMPOSE-REVIEW-003/codex/dimension'
outdir.mkdir(parents=True,exist_ok=True)

def margin(pi,q):
    best=np.full(len(pi),-np.inf)
    for eta in [1,.5,.2,.1,.05,.02,.01]:
        logits=np.log(pi)+eta*q;w=np.exp(logits-logits.max(axis=1,keepdims=True));candidate=w/w.sum(axis=1,keepdims=True)
        u=candidate-pi;den=np.abs(u).sum(axis=1)
        z=np.divide((u*q).sum(axis=1),den,out=np.full(len(pi),-np.inf),where=den>0)
        best=np.maximum(best,z)
    return float(best.max())

obs={};vars_=[];width=[];h=[];eq=[];eqfam={};h_by_fam={};ratios={};closures=[]
for rec in b['records']:
    f=rec['family'];mdp,pi,_=build_family_task(FAMILIES[f],rec['mixing'],rec['task_index'])
    d=pi.size;e=rec['routes'][b['routes'][0]]['numpy|perstate|L12']['steps'][0]
    n=np.array(e['pair_sizes']);p=np.einsum('sa,sat->st',pi,mdp['P'])
    mat=p.T-np.eye(len(pi));mat[-1]=1;rhs=np.zeros(len(pi));rhs[-1]=1
    mu=np.linalg.solve(mat,rhs)
    obs.setdefault(d,dict(family=f,nmin=[],nmean=[],kappa=[]))
    obs[d]['nmin'].append(n.min());obs[d]['nmean'].append(n.mean());obs[d]['kappa'].append((mu[:,None]*pi).min()*d)
    for route,cs in rec['routes'].items():
        curves=[]
        for name,cell in cs.items():
            v0=np.array(cell['steps'][0]['audit']['v_audit']);curve=[];cl=0
            for e in cell['steps']:
                if e['emitted']:
                    hh=margin(np.array(e['pi_before']),np.array(e['q_hat']))
                    vars_.append(float(np.median(e['pair_vars'])));width.append(e['y_range']);h.append(hh);eq.append(e['e_q'])
                    eqfam.setdefault(f,[]).append(e['e_q']);h_by_fam.setdefault(f,[]).append(hh)
                    ratios.setdefault(f,[]).append(e['e_q']/hh)
                    cl=float((np.array(e['audit']['v_after'])-v0).sum())/cell['gap_denom']
                curve.append(cl)
            curves.append(curve+[cl]*(12-len(curve)))
        closures.append(np.mean(curves,axis=0))
measured={str(d):dict(family=x['family'],tasks=len(x['nmin']),n_min_median=float(np.median(x['nmin'])),
                     n_mean_median=float(np.median(x['nmean'])),coverage_min=float(np.median(x['nmin'])/16384),
                     coverage_mean=float(np.median(x['nmean'])/16384),kappa_min_median=float(np.median(x['kappa'])),
                     headroom_over_min_visits=float(np.median(x['nmin'])/2000)) for d,x in obs.items()}
slope=math.log(measured['24']['coverage_min']/measured['12']['coverage_min'])/math.log(2)
V=float(np.median(vars_));W=float(np.median(width));H=float(np.median(h));E=float(np.median(eqfam['f1']))
anchors=dict(per_pair_variance_median=V,y_range_median=W,gate_margin_h_median=H,E_Q_median=float(np.median(eq)),
             E_Q_by_family_median={k:float(np.median(v)) for k,v in eqfam.items()},anchor_d=24,anchor_E_Q=E)
def radius(d,n):
    log=math.log(160*d)
    return math.sqrt(2*V*log/n)+7*W*log/(3*(n-1))
ref_r=radius(24,measured['24']['n_mean_median'])
def predict(d,c,exponent):
    nn=c*measured['24']['coverage_mean']*(d/24)**exponent
    return E*radius(d,nn)/ref_r
ds=[12,16,24,32,48,64,80,96,128,160,192,224,256,320,384,512,768,1000]
sens={};cost={}
for exponent in [slope,-.75,-.85,-1.]:
    label=f's={exponent:.3f}'
    sens[label]=dict(first_d_support_fails=next(d for d in ds if 16384*measured['24']['coverage_min']*(d/24)**exponent<2000),
                     first_d_median_gate_closes=next(d for d in ds if predict(d,16384,exponent)>=H),
                     chains_multiple_needed_at_d192=round(2000/(16384*measured['24']['coverage_min']*8**exponent),2))
    if exponent not in [slope,-1.]:continue
    cost[label]={}
    for d in [24,48,64,96,128,192,256,384,512,1000]:
        if predict(d,16384,exponent)<H:
            cost[label][str(d)]=dict(chains_needed=None,multiple_of_frozen=None);continue
        lo,hi=16384.,1e9
        for _ in range(100):
            mid=(lo+hi)/2
            if predict(d,mid,exponent)>H:lo=mid
            else:hi=mid
        cost[label][str(d)]=dict(chains_needed=round(hi,0),multiple_of_frozen=round(hi/16384,2))
sweep=[]
for d in [12,24,32,48,64,80,96,128,160,192,256,384,512,768,1000]:
    cov=measured['24']['coverage_min']*(d/24)**slope
    nmean=16384*measured['24']['coverage_mean']*(d/24)**slope
    hold=measured['24']['n_mean_median']/(measured['24']['coverage_mean']*(d/24)**slope)
    pred=predict(d,16384,slope)
    sweep.append(dict(d=d,coverage_min_predicted=round(cov,5),n_min_predicted=round(16384*cov,0),
                     n_mean_predicted=round(nmean,0),support_ok_at_frozen_chains=bool(16384*cov>=2000),
                     chains_needed_for_support=round(2000/cov,0),chains_needed_to_hold_per_pair_n=round(hold,0),
                     E_Q_anchored_prediction=round(pred,5),E_Q_over_h_anchored=round(pred/H,4),
                     median_gate_would_close=bool(pred>=H),batch_cost_multiple=round(hold/16384,2)))
top_scalars=dict(first_d_in_sweep_where_support_fails_at_frozen_chains=next(x['d'] for x in sweep if not x['support_ok_at_frozen_chains']),
                 first_d_in_sweep_where_median_gate_closes_at_frozen_chains=next(x['d'] for x in sweep if x['median_gate_would_close']))
diffs=[]
def compare(a,z,path):
    if isinstance(a,dict):
        for k,v in a.items():compare(v,z[k],path+'/'+k)
    elif isinstance(a,list):
        if len(a)!=len(z):diffs.append([path,'length',len(a),len(z)])
        for i,(v,w) in enumerate(zip(a,z)):compare(v,w,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if abs(a-z)>1e-10:diffs.append([path,a,z])
    elif a!=z:diffs.append([path,a,z])
for name,obj in [('measured',measured),('anchors',anchors),('exponent_sensitivity',sens),('chains_to_keep_gate_open',cost),('sweep',sweep)]:compare(obj,reference[name],name)
for name,val in top_scalars.items():compare(val,reference[name],name)
compare(slope,reference['coverage_power_law']['slope_in_d'],'slope')
a=np.array(closures);means=a.mean(axis=0)
out=dict(verdict='PASS' if not diffs else 'FAIL',differences=diffs,anchors=anchors,measured=measured,slope=slope,
         sensitivity=sens,cost=cost,sweep=sweep,top_scalars=top_scalars,
         recomputed_closure=means.tolist(),tail_k6_12_points=float(100*(means[11]-means[4])),
         tail_k5_12_points=float(100*(means[11]-means[3])),
         positive_increment_counts=(np.diff(np.column_stack([np.zeros(len(a)),a]),axis=1)>0).sum(axis=0).tolist(),
         anchor_heterogeneity=dict(pooled_median_h=H,family_median_h={k:float(np.median(v)) for k,v in h_by_fam.items()},
            family_median_eq_over_h={k:float(np.median(v)) for k,v in ratios.items()},synthetic_anchor_eq_over_h=E/H),
         source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'formal/task_results.json',BASE/'verification/dimension_scaling.json',Path(__file__)]},
         limitation='PASS means reproducing the explicit two-point calibrated arithmetic, not validating dimension scaling, bounds, median stopping probability, or runtime.')
(outdir/'result.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['verdict','differences','slope','sensitivity','anchor_heterogeneity','tail_k6_12_points','tail_k5_12_points']},indent=2))
