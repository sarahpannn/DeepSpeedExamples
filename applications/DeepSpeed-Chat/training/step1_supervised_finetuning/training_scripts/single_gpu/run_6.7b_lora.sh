#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path /home/vladislavlialin/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 6 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 192 \
   --gradient_checkpointing \
   --zero_stage 0 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log
