"""Independent complete metadata, value, H0, and cost audit complementing live replay."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parents[4];sys.path.insert(0,str(P))
from evaluate_fp_xfam_001 import FAMILIES,build_family_task
H=Path('C:/Users/Admin/Desktop/research/icrl_softmax')
paths=[H/f'results/FP-COMPOSE-00{k}/claude/formal/task_results.json' for k in [1,2]]
old,b=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
key=lambda r:(r['family'],r['mixing'],r['task_index'])
om={key(r):r for r in old['records']}; failures=[];count=0;maxerr={};unique=set();items={};forwards=0
def close(x,y,t,label):
    error=float(np.max(np.abs(np.array(x)-np.array(y))));maxerr[label]=max(maxerr.get(label,0.),error)
    if not np.isfinite(error) or error>t:failures.append([loc,label,error])
def expect(x,label):
    if not x:failures.append([loc,label])
for r in b['records']:
    loc=key(r);fam=FAMILIES[r['family']];mdp,beta,_=build_family_task(fam,r['mixing'],r['task_index'])
    kernel=np.array(mdp['P']);rewards=np.sum(kernel*mdp['R'],axis=2);g=fam['gamma'];S,A=beta.shape
    def value(pi):
        kp=(pi[:,:,None]*kernel).sum(axis=1)
        return np.linalg.solve(np.eye(S)-g*kp,(pi*rewards).sum(axis=1))
    v0=value(beta);greedy=beta.copy()
    for _ in range(100):
        v=value(greedy);q=rewards+g*(kernel*v[None,None,:]).sum(axis=2)
        choice=np.eye(A)[q.argmax(axis=1)]
        if np.array_equal(greedy,choice):break
        greedy=choice
    vstar=value(greedy);denom=(vstar-v0).sum()
    expect(np.abs(mdp['R']).max()<=fam['reward_bound'],'reward_bound')
    close(kernel.sum(axis=2),np.ones((S,A)),1e-14,'P_row_sums')
    for rt,cs in r['routes'].items():
        for name,c in cs.items():
            close(c['gap_denom'],denom,1e-10,'gap_denom')
            for i,e in enumerate(c['steps']):
                loc=(*key(r),rt,name,i+1)
                if i<4:
                    prev=om[key(r)]['routes'][rt][name]['steps'][i]
                    differing=[f for f in prev if f!='delta_total' and prev[f]!=e[f]]
                    expect(not differing,['all_H0_fields_except_declared_delta_total',differing]);count+=1
                expect(e['step']==i+1,'step_index');expect(e['delta_step']==.0125,'delta_step')
                close(e['delta_total'],.15,1e-15,'delta_total');close(e['delta_dir'],.0125/(S*A),0,'delta_dir')
                pi=np.array(e['pi_before']);qhat=np.array(e['q_hat']);v=value(pi)
                close(v,e['audit']['v_audit'],1e-10,'v_audit')
                truth=rewards+g*(kernel*v[None,None,:]).sum(axis=2)
                actual=float(np.max(np.abs(qhat-truth)));safety=e['e_q']-actual
                close(actual,e['audit']['realized_sup_error'],1e-10,'realized_sup_error')
                close(safety,e['audit']['safety_margin'],1e-10,'safety_margin')
                expect(e['audit']['covers']==(safety>=0),'covers')
                expected_reasons=[] if e['emitted'] else ['improvement_lcb_nonpositive']
                expect(e['ordered_reasons']==expected_reasons,'ordered_reasons')
                chosen=[]
                for s,eta in enumerate(e['rows_eta']):
                    if eta is not None:
                        logits=np.log(pi)+eta*qhat;w=np.exp(logits-logits.max(axis=1,keepdims=True));candidate=w/w.sum(axis=1,keepdims=True)
                        u=candidate[s]-pi[s];chosen.append(float((u*qhat[s]).sum()-e['e_q']*np.abs(u).sum()))
                if chosen:close(min(chosen),e['audit']['min_chosen_lb_over_updated'],1e-10,'min_chosen_lb')
                else:expect(e['audit']['min_chosen_lb_over_updated'] is None,'empty_min_chosen_lb')
                close(pi.sum(axis=1),np.ones(S),1e-14,'pi_row_sums');expect(np.min(pi)>0,'pi_positive')
                unique.add((*key(r),i+1));items[name]=items.get(name,0)+e['items_this_step']
                expect(e['items_this_step']==1048576,'items_this_step')
                forwards+=int(name.startswith('network'))
                if e['emitted']:
                    va=value(np.array(e['pi_after']));dv=va-v
                    expect(e['audit']['componentwise_nondegrading']==bool(dv.min()>=0),'componentwise_nondegrading')
                    close(va,e['audit']['v_after'],1e-10,'v_after')
                    close(dv.sum(),e['audit']['total_value_gain'],1e-10,'total_value_gain')
                    close(dv.sum()/denom,e['audit']['closure_fraction'],1e-10,'closure_fraction')
            expect(c['steps'][0]['pi_before']==beta.tolist(),'initial_policy')
loc='totals'
expect(count==1536,'001_steps');expect(len(unique)==b['cost']['unique_batches'],'unique_batches')
expect(len(unique)*1048576==b['cost']['items_unique'],'unique_items');expect(forwards==b['net_forwards'],'network_forwards')
expect(items==b['cost']['items_if_run_alone'],'standalone_costs')
out=dict(verdict='PASS' if not failures else 'FAIL',failures=failures,H0_steps=count,max_errors=maxerr,
         unique_batches=len(unique),network_forwards=forwards,items_if_run_alone=items,
         input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]})
outp=P/'results/FP-COMPOSE-REVIEW-003/codex/supplement.json';outp.parent.mkdir(parents=True,exist_ok=True)
outp.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
