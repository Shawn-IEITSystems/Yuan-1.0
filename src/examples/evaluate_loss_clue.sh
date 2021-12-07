# run test_generate_gpt

VOCAB_FILE=vocab.txt
DATA_PATH="../clue_data/bustm_few/dev.json"
TASK=bustm

CHECKPOINT_PATH=../yuan_checkpoints/yuan_10B/

OUTPUT_FILE=./output/$TASK'_output.json'

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6000 \
  tools/generate_loss_gpt.py \
  --greedy \
  --task $TASK \
  --sample-input-file $DATA_PATH \
  --sample-class-file "" \
  --sample-output-file $OUTPUT_FILE \
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
  --recompute \
  --fp16 \
  --encoder-attn-mask-type padding \
  --reset-position-ids \
  --reset-attention-mask \
  --eod-mask-loss \
  --out-seq-length 2 \
  --top_k 5

