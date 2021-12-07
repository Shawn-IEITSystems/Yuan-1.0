# run to generate text

VOCAB_FILE=vocab.txt

CHECKPOINT_PATH=/workspace/yuan_checkpoints/yuan_10B/

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6000 \
  tools/generate_samples_gpt.py \
  --sample-output-file "input/poetry_extreme_output.txt" \
  --sample-input-file "input/poetry_extreme_input.txt" \
  --load $CHECKPOINT_PATH \
  --tokenizer-type EncDecTokenizer \
  --vocab-file $VOCAB_FILE  \
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
  --out-seq-length 50 \
  --top_k 5 \
  --top_p 0.8 \
  --temperature 1.0
