#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH -J cao_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --qos=titan
#SBATCH --nodes=1

set -euo pipefail

# =========================
# 计时工具
# =========================
stage_start() {
	STAGE_NAME="$1"
	START_TIME=$(date +%s)
	echo ""
	echo "=============================================="
	echo " START STAGE: ${STAGE_NAME}"
	echo " START TIME : $(date '+%Y-%m-%d %H:%M:%S')"
	echo "=============================================="
}

stage_end() {
	END_TIME=$(date +%s)
	DURATION=$((END_TIME - START_TIME))
	H=$((DURATION / 3600))
	M=$(((DURATION % 3600) / 60))
	S=$((DURATION % 60))

	echo ""
	echo "=============================================="
	echo " END STAGE: ${STAGE_NAME}"
	echo " END TIME  : $(date '+%Y-%m-%d %H:%M:%S')"
	printf " DURATION  : %02d:%02d:%02d (hh:mm:ss)\n" $H $M $S
	echo "=============================================="
	echo ""
}

# =========================
# 基本信息
# =========================
echo "当前目录: $(pwd)"
echo "任务开始时间: $(date)"
echo "运行节点: $(hostname)"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# =========================
# Conda / venv
# =========================
CONDA_BASE="/home/xuyang_lab/cse12212752/miniconda3"
PROJECT_ENV="/home/xuyang_lab/cse12212752/project/NLP-project/.venv"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PROJECT_ENV"

echo "==================== 环境信息 ===================="
echo "Python路径: $(which python)"
python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available())"

# =========================
# 路径与模型
# =========================
PROJECT_DIR="/home/xuyang_lab/cse12212752/project/NLP-project"
CODE_DIR="$PROJECT_DIR/code"

export HF_HOME=${HF_HOME:-"$PROJECT_DIR/.cache/huggingface"}
MODEL_T5_BASE="${PROJECT_DIR}/t5-base"
MODEL_T5_LARGE="${PROJECT_DIR}/t5-large"

INIT_MODEL_ARG=""
SCORER_MODEL_ARG=""
FINAL_MODEL_ARG=""

[ -d "$MODEL_T5_BASE" ] && INIT_MODEL_ARG="-m $MODEL_T5_BASE" && FINAL_MODEL_ARG="-m $MODEL_T5_BASE"
[ -d "$MODEL_T5_LARGE" ] && SCORER_MODEL_ARG="-m $MODEL_T5_LARGE"

cd "$CODE_DIR"
echo "工作目录: $(pwd)"
chmod +x bash/* || true

# =========================
# 各阶段定义
# =========================
run_init(){
	stage_start "Train Initial ASQP Model (GAS Baseline)"
	bash/train_quad.sh -c 0 -d acos/rest16 -b quad -s 42 ${INIT_MODEL_ARG}
	stage_end
}

run_pseudo(){
	stage_start "Pseudo Labeling (SKIPPED)"
	echo "⚠️  Skipping pseudo-label generation (Yelp raw data not available)"
	stage_end
}

run_scorer(){
	stage_start "Train Scorer (T5-large)"
	bash/train_scorer.sh -c 0 -d acos/rest16 -b scorer -s 42 -l 20 -t 01234+ -a 1 ${SCORER_MODEL_ARG}
	stage_end
}

run_filter(){
	stage_start "Filter Pseudo Labels"
	bash/do_filtering.sh -c 0 -d acos/rest16 -b scorer
	stage_end
}

run_final(){
	stage_start "Joint Training (CS-FILTER)"
	bash/train_quad.sh -c 0 -d acos/rest16 -b 10-40_10000 \
		-f 10-40_10000 \
		-t ../output/filter/acos/rest16.json \
		${FINAL_MODEL_ARG}
	stage_end
}

run_rerank(){
	stage_start "Re-ranking"
	bash/do_reranking.sh -c 0 -d acos/rest16 -b scorer -q 10-40_10000 -a 2024-6-21
	stage_end
}

# =========================
# 入口控制
# =========================
STAGE=${1:-}

case "$STAGE" in
	init)
		run_init
		;;
	pseudo)
		run_pseudo
		;;
	scorer)
		run_scorer
		;;
	filter)
		run_filter
		;;
	final)
		run_final
		;;
	rerank)
		run_rerank
		;;
	all)
		run_init
		run_pseudo
		run_scorer
		run_filter
		run_final
		run_rerank
		;;
	"")
		echo "用法: sbatch run_pipeline.sh {init|scorer|filter|final|rerank|all}"
		exit 1
		;;
	*)
		echo "未知的 stage: $STAGE"
		exit 2
		;;
esac

echo "任务结束时间: $(date)"