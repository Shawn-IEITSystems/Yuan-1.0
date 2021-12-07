# run test_generate_gpt
NNODES=1
GPUS_PER_NODE=8
MASTER_PORT=12306
VOCAB_FILE=vocab.txt

CKPTNAME=gpt3_case11_merged_4way
CHECKPOINT_PATH=/mnt/inspur/NLP/$CKPTNAME

TASK_TYPE=eprstmt
PCF=test_pcf.txt
CURPATH=$(pwd)
OUTPATH=$CURPATH/tasks_results/$CKPTNAME/$TASK_TYPE
mkdir -p $OUTPATH

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6902 \
  tools/generate_logits_gpt.py \
  --sample-input-file "/mnt/inspur/cluedata/generate_logits/$TASK_TYPE/test_sentences.txt" \
  --sample-class-file "/mnt/inspur/cluedata/generate_logits/$TASK_TYPE/labels.json" \
  --sample-output-file "$OUTPATH/output_logits.txt" \
  --load $CHECKPOINT_PATH \
  --tokenizer-type EncDecTokenizer \
  --vocab-file $VOCAB_FILE \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --num-layers 40 \
  --hidden-size 5120 \
  --num-attention-heads 40 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --DDP-impl local \
  --encoder-attn-mask-type padding \
  --eod-mask-loss \
  --out-seq-length 2 \
  --top_k 0 \
  --top_p 0.9 \
  --temperature 0.9 \
  --fp16 \
  --recompute 
#  #--greedy \
#  #--recompute \
#  #--reset-position-ids \
#  #--reset-attention-mask \
#

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6902 \
  tools/generate_logits_gpt.py \
  --sample-input-file "/mnt/inspur/cluedata/generate_logits/$TASK_TYPE/$PCF" \
  --sample-class-file "/mnt/inspur/cluedata/generate_logits/$TASK_TYPE/labels.json" \
  --sample-output-file "$OUTPATH/output_logits_pcf.txt" \
  --load $CHECKPOINT_PATH \
  --tokenizer-type EncDecTokenizer \
  --vocab-file $VOCAB_FILE \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 2 \
  --num-layers 40 \
  --hidden-size 5120 \
  --num-attention-heads 40 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --DDP-impl local \
  --encoder-attn-mask-type padding \
  --eod-mask-loss \
  --out-seq-length 2 \
  --top_k 0 \
  --top_p 0.9 \
  --temperature 0.9 \
  --recompute \
  --fp16 \
  --zero-shot 'Calibration' 



python tools/generate_eval.py \
  --task $TASK_TYPE \
  --data_path "/mnt/inspur/cluedata/generate_logits/" \
  --output_path $OUTPATH

