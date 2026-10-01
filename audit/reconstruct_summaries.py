"""Reconstruct all retained sequential summaries and audit their decisions."""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import math
import numpy as np
from audit_seq_lib import derive_wikitext_summary, derive_task_summary, derive_mc2_summary
from summary_compare import ReductionBudget, compare_summary, verify_hash

HERE = Path(__file__).resolve().parent


def budget_for(seq, blk, lengths=None, ncands=None):
    a,b = np.asarray(seq,dtype=float),np.asarray(blk,dtype=float)
    lengths = np.ones(1,dtype=int) if lengths is None else lengths
    return ReductionBudget(tokens=int(a.size), items=int(a.shape[0] if ncands is None else len(ncands)),
        token_scale=max(float(np.abs(a).max()),float(np.abs(b).max())),
        mean_scale=max(float(np.abs(a).mean()),float(np.abs(b).mean())),
        candidate_length=int(np.max(lengths)),
        candidate_scale=float(np.max(lengths))*max(float(np.abs(a).max()),float(np.abs(b).max())),
        choices=1 if ncands is None else int(np.max(ncands)))


def summaries(raw, res):
    for tag in ('soc110m5','soc110m1'):
        a,b = raw[tag+'_wikitext_seq_lp'], raw[tag+'_wikitext_blk_lp']
        yield tag+'/wikitext', res[tag]['s1_heldout']['summary'], derive_wikitext_summary(a,b), budget_for(a,b)
        tasks=dict(res[tag]['s2_benchmarks']);tasks.update(res[tag].get('s2_auxiliary',{}))
        for task,rec in tasks.items():
            prefix=tag+'_'+task
            a,b = raw[prefix+'_seq_flat'],raw[prefix+'_blk_flat']
            lens,n = raw[prefix+'_cand_lens'],raw[prefix+'_item_ncands']
            is_mc2 = prefix+'_labels' in raw
            target=raw[prefix+('_labels' if is_mc2 else '_gold')]
            chars=raw[prefix+'_char_lens']
            fn = derive_mc2_summary if is_mc2 else derive_task_summary
            yield tag+'/'+task,rec['summary'],json.loads(json.dumps(fn(a,b,lens,n,target,chars), default=lambda x: x.item())),budget_for(a,b,lens,n)


def decision_record(raw):
    """Independently recompute per-item ties, ranks, margin gates and signs.
    math.fsum supplies a different summation path from the producer's np.sum.
    The stored manifest binds individual decisions, not only their counts.
    """
    out={}
    for tag in ('soc110m5','soc110m1'):
        for task in ('arc_easy','arc_challenge','hellaswag','piqa','openbookqa',
                     'social_iqa','winogrande','truthfulqa_mc1','truthfulqa_mc2'):
            p=tag+'_'+task;lens=raw[p+'_cand_lens'];ns=raw[p+'_item_ncands'];ch=raw[p+'_char_lens']
            edges=np.r_[0,np.cumsum(lens)];ie=np.r_[0,np.cumsum(ns)]
            scores=[]
            for side in ('blk','seq'):
                vals=raw[p+'_'+side+'_flat']
                scores.append(np.array([math.fsum(map(float,vals[a:b])) for a,b in zip(edges[:-1],edges[1:])]))
            decisions=[]
            for a,b in zip(ie[:-1],ie[1:]):
                block,seq=(v[a:b] for v in scores)
                norm=(block/ch[a:b],seq/ch[a:b])
                ranks=[]
                for v in (block,seq,*norm):
                    ranks.append([int(i) for i in np.flatnonzero(v==max(v))])
                pairs=[[int(np.sign(block[i]-block[j])),int(np.sign(seq[i]-seq[j]))]
                       for i in range(b-a) for j in range(i+1,b-a)]
                top=sorted(block,reverse=True);margin=top[0]-top[1]
                decisions.append({'maximizers':ranks,'pair_signs':pairs,
                    'gap_above_half_margin':bool(max(abs(seq-block))>=margin/2)})
            out[tag+'/'+task]=decisions
        gap=raw[tag+'_wikitext_seq_lp']-raw[tag+'_wikitext_blk_lp']
        out[tag+'/wikitext-signs-sha256']=hashlib.sha256(np.sign(gap).astype('i1').tobytes()).hexdigest()
    return out


def decimal_mc2(raw,tag):
    """80-digit reference for the cancellation-dominated probability gap.
    Candidate totals and shifted exponentials are computed in Decimal.
    """
    p=tag+'_truthfulqa_mc2';lens=raw[p+'_cand_lens'];ns=raw[p+'_item_ncands'];labels=raw[p+'_labels']
    edges=np.r_[0,np.cumsum(lens)];ie=np.r_[0,np.cumsum(ns)]
    with localcontext() as ctx:
        ctx.prec=80
        sides=[]
        for side in ('blk','seq'):
            vals=raw[p+'_'+side+'_flat']
            totals=[sum((Decimal.from_float(float(x)) for x in vals[a:b]),Decimal(0)) for a,b in zip(edges[:-1],edges[1:])]
            masses=[]
            for a,b in zip(ie[:-1],ie[1:]):
                ll=totals[a:b];top=max(ll);e=[(x-top).exp() for x in ll]
                masses.append(sum((x for x,l in zip(e,labels[a:b]) if l),Decimal(0))/sum(e))
            sides.append(masses)
        gaps=[s-b for s,b in zip(sides[1],sides[0])]
        ordered=sorted(gaps);n=len(gaps)
        return {'mc2_block_mean':float(sum(sides[0])/n),'mc2_sequential_mean':float(sum(sides[1])/n),
                'mc2_mean_gap':float(sum(gaps)/n),
                'per_item_metric_signed':{'median':float((ordered[(n-1)//2]+ordered[n//2])/2),
                                          'mean':float(sum(gaps)/n)}}


def verify_decisions(expected, actual):
    if json.dumps(expected,sort_keys=True,allow_nan=False) != json.dumps(actual,sort_keys=True,allow_nan=False):
        raise AssertionError('per-item scientific decision changed')


def run():
    res=json.loads((HERE/'audit_seq_results.json').read_text())
    pins=json.loads((HERE/'retained-inputs.json').read_text())
    for name,sha in pins['files'].items():verify_hash(HERE/name,sha)
    rows=[];references=[]
    with np.load(HERE/'audit_seq_raw.npz',allow_pickle=False) as raw:
        for name,expected,actual,budget in summaries(raw,res):
            records=compare_summary(expected,actual,budget)
            rows.extend(dict(summary=name,**r) for r in records)
        decisions=decision_record(raw)
        verify_decisions(json.loads((HERE/'sequential-decisions.json').read_text()), decisions)
        for tag in ('soc110m5','soc110m1'):
            p=tag+'_truthfulqa_mc2';ref=decimal_mc2(raw,tag)
            actual=res[tag]['s2_benchmarks']['truthfulqa_mc2']['summary']
            selected={k:actual[k] for k in ref}
            budget=budget_for(raw[p+'_seq_flat'],raw[p+'_blk_flat'],raw[p+'_cand_lens'],raw[p+'_item_ncands'])
            references.extend(dict(summary=tag,**r) for r in compare_summary(selected,ref,budget))
    classes={}
    for row in rows:
        rec=classes.setdefault(row['quantity'],dict(fields=0,max_absolute_error=0.0,max_limit=0.0))
        rec['fields']+=1;rec['max_absolute_error']=max(rec['max_absolute_error'],row['absolute_error']);rec['max_limit']=max(rec['max_limit'],row['limit'])
    return dict(status='passed',numpy=np.__version__,derived_float_fields=len(rows),
                quantity_classes=classes,all_printed_9_digits_unchanged=all(r['printed_9_digits_unchanged'] for r in rows),
                changed_fields=[r for r in rows if r['absolute_error']],decimal_reference=references,
                exact_input_files=len(pins['files']),decision_manifest='sequential-decisions.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);a=parser.parse_args()
    report=run();text=json.dumps(report,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text)
