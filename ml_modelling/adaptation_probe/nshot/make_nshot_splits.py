"""N-shot record-level splits on the MonoAlg3D/Smith2026 external set, on a FIXED test set.

Deterministic, stratified by disease class, split by RECORD (never by unit):
  per class (20 records): 10 -> permanent TEST (fixed for zero-shot AND every few-shot point),
                           2  -> validation (early stopping), 8 -> adaptation train pool.
For N in {1,2,5,8} we draw the first N train records/class; the test and val sets never change,
so zero-shot (N=0) and every few-shot N are scored on the identical 600-unit test set.
"""
import csv, collections
SRC='/mnt/user-data/uploads/internal/ECG_Delineation/test_export/smith2026_test_units_qrsanchor.csv'
rows=list(csv.DictReader(open(SRC))); fn=list(rows[0].keys())
by=collections.OrderedDict()
for r in rows: by.setdefault(r['record_id'],[]).append(r)
cls=collections.defaultdict(list)
for rec,rs in by.items(): cls[rs[0]['disease_class']].append(rec)
role={}  # record -> test/val/pool
for c in sorted(cls):
    for i,rec in enumerate(sorted(cls[c])):
        role[rec]='test' if i<10 else ('val' if i<12 else 'pool')
def write(path, picker):
    out=[]
    for rec,rs in by.items():
        s=picker(rec)
        if s:
            for r in rs: r=dict(r); r['split']=s; out.append(r)
    csv.DictWriter(open(path,'w',newline=''),fieldnames=fn).writerows([dict(zip(fn,fn))]+out) if False else None
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(out)
    return len(out)
# fixed test set
nt=write('/tmp/eval/nshot_test.csv', lambda rec: 'test' if role[rec]=='test' else None)
# per-N adaptation tables (train = first N pool records/class, split label 'train'; val kept)
pool_by_class={c:[r for r in sorted(cls[c]) if role[r]=='pool'] for c in cls}
val_recs={r for r in role if role[r]=='val'}
for N in [1,2,5,8]:
    keepN={r for c in pool_by_class for r in pool_by_class[c][:N]}
    def pick(rec, keepN=keepN):
        if rec in keepN: return 'train'
        if rec in val_recs: return 'val'
        return None
    n=write(f'/tmp/eval/nshot_adapt_N{N}.csv', pick)
    print(f"N={N}: adapt table {n} units  ({N} train + 2 val records/class)")
print(f"fixed TEST: {nt} units (10 records/class)")
# integrity
tr_all=set(); 
for N in [1,2,5,8]:
    tr={r['record_id'] for r in csv.DictReader(open(f'/tmp/eval/nshot_adapt_N{N}.csv')) if r['split']=='train'}
    te={r['record_id'] for r in csv.DictReader(open('/tmp/eval/nshot_test.csv'))}
    assert not (tr & te), f"LEAK N={N}"
print("leakage check: no train record overlaps the fixed test set, all N  [OK]")
