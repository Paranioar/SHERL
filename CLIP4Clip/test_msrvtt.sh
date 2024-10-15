DATA_PATH='datasets/MSRVTT'
SIM_ENC='meanP'
SAVE_NAME=msrvtt_${SIM_ENC}_sherl
CLIP_NAME=ViT-B/32
EPOCH=2

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 9989 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--output_dir ckpts/test_msrvtt \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header ${SIM_ENC} \
--pretrained_clip_name ${CLIP_NAME} \
--init_model ckpts/${SAVE_NAME}/pytorch_model.bin.${EPOCH}