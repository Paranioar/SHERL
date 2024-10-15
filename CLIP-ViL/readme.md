
## Intro
This is code and checkpoints for the vision-and-language pre-training model in our paper "How Much Can CLIP Benefit Vision-and-Language Tasks?" ([Link](https://arxiv.org/abs/2107.06383)).

CLIP-ViL with pre-training sets new single-model state-of-the-arts on benchmarks such as VQA v2.0 (76.70 on test-std).

The code is adopted from both the [CLIP](https://github.com/openai/CLIP) repo and the [LXMERT](https://github.com/airsplay/lxmert) repo. Many thanks to the authors of these repos~


## Data & Files Required

### Image Data
We will use COCO images and Visual Genome images for pre-training. We will also use COCO images for VQA fine-tuning.

1. Download COCO images:
    ```bash
    mkdir -p data/mscoco
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/test2015.zip -P data/mscoco

    unzip data/mscoco/train2014.zip -d data/mscoco/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/ && rm data/mscoco/val2014.zip
    unzip data/mscoco/test2015.zip -d data/mscoco/ && rm data/mscoco/test2015.zip
    ```

2. Download Visual Genome images:
    ```bash
    cd clip_vl
    mkdir -p data/vg_raw_images/VG_100K/
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P data/vg_raw_images
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P data/vg_raw_images
    unzip data/vg_raw_images/images.zip -d data/vg_raw_images/VG_100K/
    unzip data/vg_raw_images/images2.zip -d data/vg_raw_images/VG_100K/
    ```

3. Download Images Width and Height [Data](https://drive.google.com/file/d/1i7NbLQ-j3edv3zjFwX7paudkYPPtUdy4/view?usp=sharing) and save as `clip_vl/data/mscoco/width_heigths.json`


### Annotation files

1. Download the pre-trainin caption files from [LXMERT](https://github.com/airsplay/lxmert):

    ```bash
    cd clip_vl
    mkdir -p data/lxmert
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

2. Download the VQA annotation files from [LXMERT](https://github.com/airsplay/lxmert):

    ```bash
    cd clip_vl
    mkdir -p data/vqa
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
    wget nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
    ```


## Environment Setup
I recommend using docker to run the experiments. Use the image `pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel` as a start. 

```bash
pip install yacs easydict pycocotools matplotlib pillow commentjson attrdict boto3 h5py requests scikit-learn ftfy regex tqdm ml_collections transformers==3.3.1 msgpack lz4 msgpack_numpy lmdb

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.1
conda install --yes -c eumetsat expect

apt-get update
apt-get install wget vim git

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Pre-Training
We follow the LXMERT to do two stage pre-training. At state one, we train without the QA loss; at stage two, we traing with the QA loss.

The general command to run experiments are:
```bash
bash BASH_FILE GPUS EXPERIMENT_NAME PORT NUM_NODES AnyOtherParameters
```

All experiments listed below are run on 8 Nvidia A100 GPUs, each with 40GB memory. 

Caveats: 
To reduce CPU memory cost, we use shared memory to share annotation files across data readers. Be sure to delete any file with the prefix `sharearray_` under `/dev/shm/` after you finish training.

### Reproduce the RN50 model pre-training
1. Command to run first-stage pre-training for RN50 model:
    ```bash
    bash scripts/pretrain.bash 0,1,2,3,4,5,6,7 clip_rn50_stage_one 9590 8 --fp16 --gradient_accumulation_steps 2 --batchSize 32 --lr 1e-4 --aspect_ratio_group_factor 5 --add_zero_padding --compress_data --warmup_ratio 0.025 --report_step 200 --numWorkers 20 --train mscoco_train,mscoco_nominival,vgnococo --epochs 20 --sub_sampling --sub_feat_num 100 --schedule 12,17 --use_separate_optimizer_for_visual --sgd_lr 0.003 --sgd_momentum 0.0 --use_positional_embedding
    ```
    When the model trains after Epoch 9, stop the training. There should be a file named `snap/clip_rn50_stage_one/Epoch09_LXRT.pth`.

2. Command to run second-stage pre-training for RN50 model:
    ```bash
    bash scripts/pretrain.bash 0,1,2,3,4,5,6,7 clip_rn50_stage_two 9590 8 --fp16 --gradient_accumulation_steps 2 --batchSize 32 --lr 5e-5 --aspect_ratio_group_factor 5 --add_zero_padding --compress_data --warmup_ratio 0.025 --report_step 200 --numWorkers 20 --train mscoco_train,mscoco_nominival,vgnococo --epochs 11 --sub_sampling --sub_feat_num 100 --schedule 4,8 --use_separate_optimizer_for_visual --sgd_lr 0.003 --sgd_momentum 0.0 --use_positional_embedding --load snap/pretrain/clip_rn50_stage_one/Epoch09 --not_load_scheduler --taskQA --not_load_adam_optimizer
    ```
    When the model finishes training, there should be a file named `snap/clip_rn50_stage_two/Epoch11_LXRT.pth`. Checkpoint on [Google Drive](https://drive.google.com/file/d/1wi57oVOCP6gyf0MoRfu-hoOyyqLSHndA/view?usp=sharing).


### Reproduce the RN50x4 model pre-training
1. Command to run first-stage pre-training for RN50x4 model:
    ```bash
    bash scripts/pretrain.bash 0,1,2,3,4,5,6,7 clip_rn50x4_stage_one 9590 8 --fp16 --gradient_accumulation_steps 2 --batchSize 30 --lr 5e-5 --aspect_ratio_group_factor 5 --add_zero_padding --compress_data --warmup_ratio 0.025 --report_step 200 --numWorkers 20 --train mscoco_train,mscoco_nominival,vgnococo --epochs 20 --sub_sampling --sub_feat_num 100 --schedule 12,17 --use_separate_optimizer_for_visual --sgd_lr 0.003 --sgd_momentum 0.0 --use_positional_embedding --clip_model_name RN50x4
    ```
    When the model trains after Epoch 9, stop the training. There should be a file named `snap/pretrain/clip_rn50x4_stage_one/Epoch09_LXRT.pth`.


2. Command to run second-stage pre-training for RN50 model:
    ```bash
    bash scripts/pretrain.bash 0,1,2,3,4,5,6,7 clip_rn50x4_stage_two 9590 8 --fp16 --gradient_accumulation_steps 2 --batchSize 30 --lr 2.75e-5 --aspect_ratio_group_factor 5 --add_zero_padding --compress_data --warmup_ratio 0.025 --report_step 200 --numWorkers 20 --train mscoco_train,mscoco_nominival,vgnococo --epochs 11 --sub_sampling --sub_feat_num 100 --schedule 4,9 --use_separate_optimizer_for_visual --sgd_lr 0.003 --sgd_momentum 0.0 --use_positional_embedding --load snap/pretrain/clip_rn50x4_stage_one/Epoch09 --not_load_scheduler --taskQA --not_load_adam_optimizer
    ```
    When the model finishes training, there should be a file named `snap/pretrain/clip_rn50x4_stage_two/Epoch11_LXRT.pth`. Checkpoint on [Google Drive](https://drive.google.com/file/d/1cbulHZS-dDk9DpWyUhiC2SqX9BJlrdxP/view?usp=sharing).


## Fine-Tuning

Currently, we provide the scripts to fine-tune on VQA. Experiments can be run on 4 Nvidia 2080Tis.

Caveats: 
To reduce CPU memory cost, we use shared memory to share annotation files across data readers. Be sure to delete any file with the prefix `sharearray_` under `/dev/shm/` after you finish training.

1. Training (RN50x4) (First download the pre-trained checkpoint to `snap/pretrain/clip_rn50x4_stage_two/Epoch11_LXRT.pth`):
    ```bash
    ./scripts/file.sh 0,1,2,3 model_name 9590 4 2
    ```
    When the model finishes training, there should be a file named `snap/vqa/vqa_clip_rn50x4/BEST_LXRT.pth`. Checkpoint on [Google Drive](https://drive.google.com/file/d/1c1DMNRow5aNRgQVrCc6Z0p3EdVVWfiT5/view?usp=sharing).


2. Testing:
    ```bash
    ./scripts/file.sh 0 model_name 9590 1 --load snap/vqa/model_name/BEST --test minival or test
    ```
    This should give the score around 73.91 on minival (minival scores are usually 2~3 points lower than those on `test-dev`).

    Change `--test minival` to `--test test` to generate a json file `snap/vqa/test/test_predict.json`, which could submited to the leaderboard. Using the provided checkpoint should give a score close to what is reported in the paper .