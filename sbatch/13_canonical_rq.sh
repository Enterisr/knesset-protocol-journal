#!/bin/bash
#SBATCH --job-name=anlp_rq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

# Step 13 — the project research question: does LLM canonicalization fix the
# representation failure that geometric de-biasing could not?
#
# PREREQ: steps/12_joint_pipeline/outputs/canonical_gold.json must exist
#         (Colab output), then build_segments.py must have been run.

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

set -e

python steps/12_joint_pipeline/code/build_segments.py

# 1. embed the canonical variant (raw side reuses Step 07's e5.npy)
python steps/13_canonical_rq/code/embed_canonical.py

# 2. the head-to-head: intrinsic AUC + streaming-honest linking, raw vs canonical
python steps/13_canonical_rq/code/rq_eval.py

# 3. end-to-end journals — demonstration artifacts, both operating points.
#    NOTE: these are NOT an evaluation surface; with linking precision ~0.13 the
#    end-to-end journal conflates linking error with summarization error. The
#    numbers that answer the RQ come from step 2 above.
for TAG in raw canonical; do
  SEG=steps/12_joint_pipeline/outputs/segments_${TAG}.json
  [ -f "$SEG" ] || { echo "skip $TAG (no $SEG)"; continue; }
  # anti-duplication point (default): 23 recurring entries at 0.95 purity -> full journal
  python steps/12_joint_pipeline/code/run_journal.py \
      --segments "$SEG" --tag "${TAG}_fmr5" --theta 0.342
  # best-F1 point: more recurring entries (76) but 0.66 purity -> linking only
  python steps/12_joint_pipeline/code/run_journal.py \
      --segments "$SEG" --tag "${TAG}_f1"   --theta 0.156 --no_logwrite
done
