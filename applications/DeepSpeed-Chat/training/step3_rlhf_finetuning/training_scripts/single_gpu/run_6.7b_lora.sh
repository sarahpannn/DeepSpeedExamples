#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


ACTOR_ZERO_STAGE="--actor_zero_stage 0"
CRITIC_ZERO_STAGE="--critic_zero_stage 0"
ACTOR_MODEL_PATH= # Provide the ckpt path of the actor model
CRITIC_MODEL_PATH= # Provide the ckpt path of the critic model

OUTPUT="./output"

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path /home/vladislavlialin/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output\
   --critic_model_name_or_path /home/vladislavlialin/DeepSpeedExamples/applications/DeepSpeed-Chat/output/reward-models/350m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 2 \
   --per_device_mini_train_batch_size 16 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --ppo_epochs 1 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 16 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} ${OFFLOAD}\
   --actor_lora_dim 128 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --disable_actor_dropout \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
