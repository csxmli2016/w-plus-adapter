#!/bin/bash

export date_now=`date "+%Y-%m-%d"`

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file 8_gpu.json --main_process_port 25655 train_wild.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --data_json_file=None \
  --data_root_path=None \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=16 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="./experiments_stage2_${date_now}" \
  --save_steps=50 \
  --lambda_disen=2.0 \
  --lambda_preserve=1.0 \
  --stage1_pretrain='./experiments_stage1_[DATE]/checkpoint-[NUM]/wplus_adapter.bin' # you can obtain this file from pytorch_model.bin by running python script/transfer_pytorchmodel_to_wplus.py

  