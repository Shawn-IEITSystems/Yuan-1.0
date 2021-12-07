#!/bin/bash

WORLD_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="../clue_data/bustm_few/train.json"
VALID_DATA="../clue_data/bustm_few/dev.json"
TEST_DATA="../clue_data/bustm_few/test.json"
VOCAB_FILE=vocab.txt


PRETRAINED_CHECKPOINT=../yuan_checkpoints/yuan_10B/

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task BUSTM \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --test-data $TEST_DATA \
               --tokenizer-type EncDecTokenizer \
               --vocab-file $VOCAB_FILE \
               --epochs 30 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 2 \
               --num-layers 40 \
               --hidden-size 5120 \
               --num-attention-heads 40 \
               --seq-length 256 \
               --max-position-embeddings 2048 \
               --micro-batch-size 16 \
               --checkpoint-activations \
               --lr 1.0e-5 \
	       --min-lr 1.0e-6 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.3 \
               --log-interval 1 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --clip-grad 1.0 \
               --hidden-dropout 0.1 \
               --attention-dropout 0.1 \
               --fp16 \
               --initial-loss-scale 128
