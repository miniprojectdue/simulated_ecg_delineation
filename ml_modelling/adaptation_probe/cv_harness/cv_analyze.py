"""Aggregate CV few-shot results into a per-class curve (T-offset MAE vs N), across folds."""
import csv, numpy as np, collections, sys, glob, os
R='/tmp/eval/ml_modelling/results'
def cls_mae(tags):
    by=collections.defaultdict(list)
    for t in tags:
        p=f'{R}/{t}/per_unit.csv'
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            e=r.get('t_offset_err_ms','')
            if e!='': by[r['disease_class']].append(abs(float(e)))
    return {c:(np.mean(v),len(v)) for c,v in by.items()}
def show(mode, Ns):
    print(f"\n=== {mode}: T-offset MAE (ms) by class, aggregated over 4 folds (all 100 records) ===")
    classes=['Healthy','AnteriorInfarction','InferiorInfarction','AnteriorIschemia','InferiorIschemia']
    zs=cls_mae([f'cv_zeroshot_f{f}' for f in range(4)])
    hdr='  N '+''.join(f'{c[:9]:>11}' for c in classes)
    print(hdr); print('  0 '+''.join(f'{zs.get(c,(float("nan"),0))[0]:>11.1f}' for c in classes)+'   (zero-shot)')
    for N in Ns:
        m=cls_mae([f'cv_{mode}_N{N}_f{f}' for f in range(4)])
        print(f'  {N:<2}'+''.join(f'{m.get(c,(float("nan"),0))[0]:>11.1f}' for c in classes))
show('balanced',[1,2,5,8,12]); show('ischemia',[2,4,8,12])
