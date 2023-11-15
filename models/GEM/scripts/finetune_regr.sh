#!/bin/bash
#SBATCH -c 16                              # Request one core
#SBATCH -t 5-00:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=300G                         # Memory total in MiB (for all cores)
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like


module load conda3/latest
source ~/.bashrc
source activate gem
cd /n/scratch3/users/t/tod192/PaddleHelix/apps/pretrained_compound/ChemRL/GEM

# cd $(dirname $0)
# cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH
dataset="tk10"
compound_encoder_config="model_configs/geognn_l8.json"
init_model="./pretrain_models-chemrl_gem/regr.pdparams"
log_prefix="log/pretrain"
thread_num=16
echo "==> $dataset"
data_path="./chemrl_downstream_datasets/$dataset"
cached_data_path="./cached_data/$dataset"
model_config_list="model_configs/down_mlp2.json"
drop_list="0.1 0.2"
batch_size_list="4 32 258"
lrs_list="1e-3 1e-4"

echo "==> $dataset"
data_path="./chemrl_downstream_datasets/$dataset"
cached_data_path="./cached_data/$dataset"
model_config_list="model_configs/down_mlp2.json"
lrs_list="1e-3 1e-4"
drop_list="0.1 0.2"
batch_size_list="4 32 258"

for batch_size in $batch_size_list;do
	for model_config in $model_config_list; do
		for lr in $lrs_list; do
			for head_lr in $lrs_list; do
				for dropout_rate in $drop_list; do
					log_dir="$log_prefix-$dataset"
					log_file_text="lr${lr}_${head_lr}_drop${dropout_rate}_batch${batch_size}"
					log_file="$log_dir/${log_file_text}.txt"
					if [ ! -f "$log_file" ]; then
						echo "Outputs redirected to $log_file"
						mkdir -p $log_dir
						start_time=$(date +%s)
						CUDA_VISIBLE_DEVICES=0 python finetune_regr_umap.py \
								--batch_size=$batch_size \
								--max_epoch=100 \
								--dataset_name=$dataset \
								--data_path=$data_path \
								--cached_data_path=$cached_data_path \
								--split_type=scaffold \
								--compound_encoder_config=$compound_encoder_config \
								--model_config=$model_config \
								--init_model=$init_model \
								--model_dir=$root_path/output/chemrl_gem/finetune/$dataset \
								--encoder_lr=$lr \
								--head_lr=$head_lr \
								--dropout_rate=$dropout_rate >> $log_file 2>&1
						cat $log_dir/* | grep FINAL| python ana_results.py > $log_dir/final_result
						end_time=$(date +%s)
						echo "Script run time: $((end_time - start_time)) seconds." >> $log_file 2>&1
						echo "Script run time: $((end_time - start_time)) seconds."
					fi
				done
			done
		done
	done
done



