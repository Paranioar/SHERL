DATASET_NAME='f30k'
DATA_PATH='./data/'${DATASET_NAME}
WEIGHT_PATH='./data/weights'
DOWN_FACTOR=2
DECAY_FACTOR=1e-2
SAVE_NAME=sherl_${DATASET_NAME}

source /opt/conda/bin/activate /opt/conda/envs/VTR

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME}  \
  --logger_name runs/${SAVE_NAME}/log --model_name runs/${SAVE_NAME} \
  --num_epochs 25 --lr_update 15 --learning_rate 5e-4 --workers 20 --log_step 200 \
  --precomp_enc_type backbone --backbone_source wsl \
  --vse_mean_warmup_epochs 1 --input_scale_factor 2.0 --batch_size 112 \
  --downsample_factor ${DOWN_FACTOR} --decay_factor ${DECAY_FACTOR}
