#!/bin/bash
# The author's serve-intel-ar.sh, with the HF hub mounted at its host path so the snapshots' symlinks resolve, the
# n-gram table from the local FP8 snapshot (no RDMA), and an optional nsys wrapper (NSYS=1) writing to $OUT/r0.
set -euo pipefail
HUB=/home/stephen/.cache/huggingface/hub
MODEL=$HUB/models--Saren--Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN/snapshots/19f9710c8f6600a32e15fb98bc77613ee8ec369b
TABLE=$HUB/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13
OUT=${OUT:?out dir}; mkdir -p $OUT
R=/home/stephen/claude-scratch/2026-09-30-reference-vllm
NAME=${NAME:-qwen38-ref}; PORT=${PORT:-18300}; IMAGE=qwen38-flash-dgx
NSYSDIR=/opt/nvidia/nsight-systems/2025.3.2
SEQS=${SEQS:-8}; MTP=${MTP:-3}; CTX=262144; GPU_MEM=0.01; KV_BYTES=20g
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
docker rm -f $NAME >/dev/null 2>&1 || true
ENTRY=(); PRE=()
if [[ "${NSYS:-0}" == 1 ]]; then
  ENTRY=(--entrypoint $NSYSDIR/bin/nsys -v $NSYSDIR:$NSYSDIR:ro)
  PRE=(profile -t cuda --cuda-graph-trace=node --cuda-flush-interval=5000 -s none --cpuctxsw=none -o /out/r0 --force-overwrite true vllm serve)
fi
docker run -d --name $NAME --gpus all --ipc=host --shm-size 16g -p $PORT:8000 \
  -v $HUB:$HUB:ro -v $R/model_dir:/model:ro -v $R/table_dir:/ple-table:ro -v $OUT:/out "${ENTRY[@]}" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=1 -e VLLM_PLE_MMAP_PREFETCH=0 \
  -e VLLM_PLE_MMAP_MADV_RANDOM=1 -e VLLM_HIT_DEBUG=0 -e VLLM_STEP_PROFILE=0 -e VLLM_PLE_MMAP_DIR=/ple-table \
  -e VLLM_QSA_EXACT_TOPK=0 -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so \
  -e VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft_vocab_65536.npy \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=0 -e CUDA_LAUNCH_BLOCKING=0 \
  $IMAGE "${PRE[@]}" \
  /model --served-model-name qwen --host 0.0.0.0 --port 8000 --load-format fastsafetensors \
  --max-model-len $CTX --max-num-seqs $SEQS --gpu-memory-utilization $GPU_MEM \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  -cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT --no-enable-flashinfer-autotune \
  --kv-cache-dtype auto --enable-logging-iteration-details --kv-cache-memory-bytes $KV_BYTES \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"
echo "started $NAME on :$PORT nsys=${NSYS:-0}"
