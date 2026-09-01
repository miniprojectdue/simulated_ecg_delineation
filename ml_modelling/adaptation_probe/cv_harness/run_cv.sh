#!/bin/bash
# Usage: run_cv.sh <checkpoint.pt>   — runs 4-fold CV few-shot sweep, balanced + ischemia-weighted.
CKPT="${1:-/mnt/user-data/uploads/Delineation/ml_modelling/checkpoints/pretrain_no_bottleneck_attention/best.pt}"
cd /tmp/eval
E="--device cpu --set data.signal_cache=/tmp/eval/cache --set data.num_workers=0 --set eval.read_pad_ms=-1 --set data.window_source=table"
for f in 0 1 2 3; do
  python3 ml_modelling/scripts/evaluate.py --checkpoint "$CKPT" --units cv/balanced/fold${f}_test.csv --tag cv_zeroshot_f${f} $E >logs_cv_zs_f${f}.txt 2>&1 && echo "zeroshot fold$f"
done
for mode in balanced ischemia; do
  Ns=$([ "$mode" = balanced ] && echo "1 2 5 8 12" || echo "2 4 8 12")
  for f in 0 1 2 3; do for N in $Ns; do
    python3 ml_modelling/scripts/train.py --config ml_modelling/configs/adapt_ischemia.yaml \
      --set data.units_csv=cv/${mode}/fold${f}_N${N}.csv --set run.name=cv_${mode}_N${N}_f${f} \
      --set train.init_from="$CKPT" ${EPOCHS:+--set train.epochs=$EPOCHS} >logs_cv_tr_${mode}_N${N}_f${f}.txt 2>&1 \
      && python3 ml_modelling/scripts/evaluate.py --checkpoint ml_modelling/checkpoints/cv_${mode}_N${N}_f${f}/best.pt \
         --units cv/${mode}/fold${f}_test.csv --tag cv_${mode}_N${N}_f${f} $E >logs_cv_ev_${mode}_N${N}_f${f}.txt 2>&1 \
      && echo "done ${mode} N=$N fold=$f"
  done; done
done
echo CVDONE
