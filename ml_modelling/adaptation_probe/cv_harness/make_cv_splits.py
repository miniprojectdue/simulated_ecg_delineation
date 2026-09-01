"""4-fold record-level cross-validation splits on the MonoAlg3D/Smith2026 external set.

20 records/class -> 4 folds of 5 test records/class (every record tested exactly once).
Per fold: 5 test, and 15 in the adaptation pool of which 3/class are val and 12/class train-eligible.

  balanced   : N train records from EVERY class, N in {1,2,5,8,12}
  ischemia   : non-ischemia classes fixed at ANCHOR=2 records each; ischemia classes swept
               N_isch in {2,4,8,12}. Answers "with the easy classes just anchored, how many
               ischemia records close the gap".
Writes cv/<mode>/fold<f>_test.csv and cv/<mode>/fold<f>_N<n>.csv (train+val split labels).
"""
import csv, collections, os
SRC='/mnt/user-data/uploads/internal/ECG_Delineation/test_export/smith2026_test_units_qrsanchor.csv'
ISCH={'AnteriorIschemia','InferiorIschemia'}
ANCHOR=2
rows=list(csv.DictReader(open(SRC))); fn=list(rows[0].keys())
by=collections.OrderedDict()
for r in rows: by.setdefault(r['record_id'],[]).append(r)
cls=collections.defaultdict(list)
for rec,rs in by.items(): cls[rs[0]['disease_class']].append(rec)
for c in cls: cls[c]=sorted(cls[c])   # deterministic

def fold_assign(c, f):
    """return dict record-> 'test'|'val'|'pool' for class c, fold f (0..3)."""
    recs=cls[c]; test=set(recs[5*f:5*f+5]); pool=[r for r in recs if r not in test]
    val=set(pool[:3]); train_pool=pool[3:]        # 3 val, 12 train-eligible
    return test, val, train_pool

def write(path, rows_out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fn); w.writeheader(); w.writerows(rows_out)

def emit(mode, folds=4):
    Ns = [1,2,5,8,12] if mode=='balanced' else [2,4,8,12]
    for f in range(folds):
        # test table for this fold
        test_rows=[]
        assign={}
        for c in cls:
            test,val,tp = fold_assign(c,f)
            for r in test: assign[r]=('test',None)
            for r in val: assign[r]=('val',None)
            for i,r in enumerate(tp): assign[r]=('train',i)   # rank within train pool
        for rec,rs in by.items():
            if assign.get(rec,(None,))[0]=='test':
                for r in rs: r=dict(r); r['split']='test'; test_rows.append(r)
        write(f'/tmp/eval/cv/{mode}/fold{f}_test.csv', test_rows)
        # adapt tables per N
        for N in Ns:
            out=[]
            for c in cls:
                test,val,tp = fold_assign(c,f)
                if mode=='balanced': n_c=N
                else: n_c = N if c in ISCH else ANCHOR
                train=set(tp[:n_c])
                for rec in train|val:
                    for r in by[rec]:
                        r=dict(r); r['split']='train' if rec in train else 'val'; out.append(r)
            write(f'/tmp/eval/cv/{mode}/fold{f}_N{N}.csv', out)
    # integrity + summary
    print(f"[{mode}] folds={folds} Ns={Ns}")
    # leakage: union of all test == all records once; no train record in same-fold test
    allrec=set(r['record_id'] for r in rows)
    tested=set()
    for f in range(folds):
        te={r['record_id'] for r in csv.DictReader(open(f'/tmp/eval/cv/{mode}/fold{f}_test.csv'))}
        tested|=te
        for N in Ns:
            tr={r['record_id'] for r in csv.DictReader(open(f'/tmp/eval/cv/{mode}/fold{f}_N{N}.csv')) if r['split']=='train'}
            assert not (tr & te), f"LEAK {mode} fold{f} N{N}"
    assert tested==allrec, f"test coverage {len(tested)} != {len(allrec)}"
    print(f"  every record tested exactly once: {len(tested)}/{len(allrec)}  |  no train/test leak in any fold  [OK]")

emit('balanced'); emit('ischemia')
