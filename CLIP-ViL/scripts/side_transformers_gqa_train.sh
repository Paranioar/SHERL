# The name of this experiment.
name=$2
r=$5
seed=9595
# Save logs and models under snap/vqa; make backup.
output=snap/gqa/${name}_r${r}@${seed}
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP
source /opt/conda/bin/activate /opt/conda/envs/CLIP-ViL

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/gqa.py \
    --distributed \
    --train train,valid --valid testdev \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 32 --optim bert --lr 1e-4 --epochs 5 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --fp16 \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.05 \
    --report_step 400 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.001 \
    --sgd_momentum 0.0 \
    --schedule 3 \
    --use_positional_embedding \
    --pos_num 25 \
    --clip_model_name RN50x4 \
    --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4/Epoch11 \
    --loss_scale 500 \
    --use_side_transformers \
    --reduction_factor ${r} \
    --compute_time \
    --compute_memory \
    --seed ${seed} \
    ${@:6}  | tee $output/log.log
