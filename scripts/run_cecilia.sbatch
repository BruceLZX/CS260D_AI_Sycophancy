#!/usr/bin/env bash
#SBATCH --job-name=syco-cecilia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --no-requeue

set -euo pipefail
python_bin=${1:?Pass the compute-visible Python executable}
model=${2:?Pass qwen2.5-3b or llama3.2-3b}
shift 2
case "$model" in qwen2.5-3b|llama3.2-3b) ;; *) exit 2 ;; esac
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to the shared cache containing the pinned snapshots}"

cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p outputs/_scratch
exec 9>"outputs/_scratch/$model.lock"
flock -n 9
srun "$python_bin" -m syco --model-config config.cecilia.yaml generate --models "$model" "$@"
