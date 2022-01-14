#! /bin/bash

NNODES=224
GPUS_PER_NODE=8
MASTER_PORT=12306

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
LOAD_CHECKPOINT_PATH=./checkpoints/gpt3_case11_300B/
SAVE_CHECKPOINT_PATH=./checkpoints/gpt3_case11_300B/
TENSORBOARD_PATH=./tensorboard/gpt3_case11_300B/$DATETIME


VOCAB_FILE=vocab.txt
DATA_PATH=$(cat data_path_aug.txt)

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_gpt.py \
        --tokenizer-type EncDecTokenizer \
        --vocab-file $VOCAB_FILE \
        --tensor-model-parallel-size 8 \
        --pipeline-model-parallel-size 2 \
        --num-layers 40 \
        --hidden-size 5120 \
        --num-attention-heads 40 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size 2688 \
        --train-samples 146484375 \
        --rampup-batch-size 448 448 488280 \
        --lr-decay-samples 131835937 \
        --lr-warmup-samples 488280 \
        --lr 1.0e-04 \
        --min-lr 1.0e-05 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --save-interval 2000 \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --DDP-impl local \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --checkpoint-activations \
        --checkpoint-num-layers 1 \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --encoder-attn-mask-type padding   
