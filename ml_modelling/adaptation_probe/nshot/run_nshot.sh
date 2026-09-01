#!/bin/bash
# Usage: run_nshot.sh <checkpoint.pt> <tag_prefix>   (defaults to no_bottleneck as a stand-in)
CKPT="${1:-/mnt/user-data/uploads/Delineation/ml_modelling/checkpoints/pretrain_no_bottleneck_attention/best.pt}"
PREF="${2:-nshot}"
cd /tmp/eval
EVAL="--device cpu --set data.signal_cache=/tmp/eval/cache --set data.num_workers=0 --set eval.read_pad_ms=-1 --set data.window_source=table"
# zero-shot (N=0) on the fixed test
python3 ml_modelling/scripts/evaluate.py --checkpoint "$CKPT" --units /tmp/eval/nshot_test.csv --tag ${PREF}_N0 $EVAL >/tmp/eval/nlog_N0.txt 2>&1 && echo "done N=0 (zero-shot)"
for N in 1 2 5 8; do
  python3 ml_modelling/scripts/train.py --config ml_modelling/configs/adapt_ischemia.yaml \
     --set data.units_csv=/tmp/eval/nshot_adapt_N${N}.csv --set run.name=${PREF}_adapt_N${N} \
     --set train.init_from=$CKPT ${EPOCHS:+--set train.epochs=$EPOCHS} >/tmp/eval/nlog_train_N${N}.txt 2>&1 || { echo "TRAIN FAIL N=$N"; tail -5 /tmp/eval/nlog_train_N${N}.txt; continue; }
  python3 ml_modelling/scripts/evaluate.py --checkpoint ml_modelling/checkpoints/${PREF}_adapt_N${N}/best.pt \
     --units /tmp/eval/nshot_test.csv --tag ${PREF}_N${N} $EVAL >/tmp/eval/nlog_eval_N${N}.txt 2>&1 && echo "done N=$N"
done
echo NSHOTDONE
