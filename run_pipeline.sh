#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=rtx2080ti
#SBATCH -J cao_job_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --qos=rtx2080ti
#SBATCH --nodes=1

#sbatch run_pipeline.sh all   # 用这个命令顺序执行全部6个阶段
#sbatch run_pipeline.sh run_init   # 用这个命令顺序执行特定阶段，e.g run_init阶段

set -euo pipefail

# 打印当前信息
echo "当前目录: $(pwd)"
echo "任务开始时间: $(date)"
echo "运行节点: $(hostname)"

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# 先确认GPU是否分配成功
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

#source activate base

## 自动激活项目环境
CONDA_BASE="/home/xuyang_lab/cse12212752/miniconda3"
PROJECT_ENV="/home/xuyang_lab/cse12212752/project/NLP-project/.venv"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PROJECT_ENV"

echo "==================== 环境信息 ===================="
# 打印激活后的 Python 路径和版本
echo "Python路径: $(which python)"
echo "Python版本:"
python --version
echo ""

python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('current_device=', torch.cuda.current_device() if torch.cuda.is_available() else -1)"

## 修改：支持按阶段运行 README 中的步骤（init/pseudo/scorer/filter/final/rerank/all）

PROJECT_DIR="/home/xuyang_lab/cse12212752/project/NLP-project"
CODE_DIR="$PROJECT_DIR/code"

STAGE=${1:-}

cd /home/xuyang_lab/cse12212752/project/NLP-project/code

echo "工作目录: $(pwd)"

# 已在脚本开始处固定激活 PROJECT_ENV

# 确保脚本可执行
chmod +x bash/* || true

run_init(){
		echo "运行：训练初始模型"
		bash/train_quad.sh -c 0 -d acos/rest16 -b quad -s 42
}
run_pseudo(){
		echo "运行：伪标注"
		bash/pseudo_labeling.sh -c 0 -d acos/rest16 -b quad
}
run_scorer(){
		echo "运行：训练 scorer"
		bash/train_scorer.sh -c 0 -d acos/rest16 -b scorer -s 42 -l 20 -t 01234+ -a 1
}
run_filter(){
		echo "运行：过滤伪标注"
		bash/do_filtering.sh -c 0 -d acos/rest16 -b scorer
}
run_final(){
		echo "运行：使用过滤后数据训练 ASQP 模型"
		bash/train_quad.sh -c 0 -d acos/rest16 -b 10-40_10000 -f 10-40_10000 -t ../output/filter/acos/rest16.json
}
run_rerank(){
		echo "运行：重排（re-rank）"
		bash/do_reranking.sh -c 0 -d acos/rest16 -b scorer -q 10-40_10000 -a 2024-6-21
}

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
		echo "用法: sbatch myjob.sh <stage>  ，stage 可选: init|pseudo|scorer|filter|final|rerank|all"
		exit 1
		;;
	*)
		echo "未知的 stage: $STAGE"
		exit 2
		;;
esac

echo "任务结束时间: $(date)"