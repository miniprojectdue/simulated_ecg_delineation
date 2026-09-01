"""Stratified RECORD-level split of Smith2026 for few-shot domain adaptation.

Splitting by unit would put the 12 leads of one record (near-identical delineation problems
sharing one morphology) on both sides -> leakage. So we split by RECORD, stratified by disease
class, deterministically. Per class (20 records): 10 test, 8 train, 2 val.
"""
import csv, collections, sys
SRC='/mnt/user-data/uploads/internal/ECG_Delineation/test_export/smith2026_test_units_qrsanchor.csv'
rows=list(csv.DictReader(open(SRC)))
fn=list(rows[0].keys())
by_rec=collections.OrderedDict()
for r in rows: by_rec.setdefault(r['record_id'], []).append(r)
# class per record
cls_recs=collections.defaultdict(list)
for rec,rs in by_rec.items(): cls_recs[rs[0]['disease_class']].append(rec)
assign={}
for cls in sorted(cls_recs):
    recs=sorted(cls_recs[cls])              # deterministic order
    for i,rec in enumerate(recs):
        assign[rec] = 'test' if i<10 else ('val' if i<12 else 'train')
# write tables
def write(path, split_names):
    out=[]
    for rec,rs in by_rec.items():
        if assign[rec] in split_names:
            for r in rs:
                r2=dict(r); r2['split']=assign[rec]; out.append(r2)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(out)
    return len(out)
base='/tmp/eval/'
n_adapt=write(base+'smith2026_adapt.csv', {'train','val'})   # 12 records/class -> train+val
n_test =write(base+'smith2026_heldout.csv', {'test'})        # 10 records/class -> test
# report
print("class            train_rec val_rec test_rec")
tot=collections.Counter()
for cls in sorted(cls_recs):
    c=collections.Counter(assign[r] for r in cls_recs[cls]); tot+=c
    print(f"  {cls:20} {c['train']:6d} {c['val']:6d} {c['test']:7d}")
print(f"  {'TOTAL records':20} {tot['train']:6d} {tot['val']:6d} {tot['test']:7d}")
print(f"units: adapt(train+val)={n_adapt}  heldout(test)={n_test}")
# leakage check: no record in both
adapt_recs={r for r in assign if assign[r] in ('train','val')}
test_recs ={r for r in assign if assign[r]=='test'}
print("record overlap adapt∩test:", len(adapt_recs & test_recs), "(must be 0)")
