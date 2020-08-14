#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/path/to/glue/data

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=MRPC  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi

ENTROPY=0.15
echo $ENTROPY

python -um examples.run_highway_glue_prediction \
  --model_type $MODEL_TYPE \
  --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/raw \
  --task_name $DATASET \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/raw \
  --plot_data_dir ./plotting/ \
  --max_seq_length 128 \
  --early_exit_entropy $ENTROPY \
  --eval_highway \
  --overwrite_cache \
  --per_gpu_eval_batch_size=1 \
  --gamma 0.9 \
  --temper 3.0 \
  --kd_loss_type other
