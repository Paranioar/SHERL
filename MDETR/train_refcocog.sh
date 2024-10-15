source /opt/conda/bin/activate /opt/conda/envs/MDETR
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11021 --use_env main.py --output_dir runs/refcocog_sherl --dataset_config configs/refcocog.json --batch_size 4 --load pretrained_weights/pretrained_resnet101_checkpoint.pth --ema --text_encoder_lr 1e-5 --lr 5e-4
