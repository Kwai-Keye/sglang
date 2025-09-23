#!/bin/bash


if [ $# -ne 6 ]; then
    echo "Usage: ./run.sh <model_path> <dist_init_addr> <port> <node_rank> <stdout_log_path> <stderr_log_path>"
    exit 1
fi

model_path=$1
dist_init_addr=$2
port=$3
node_rank=$4
stdout_log_path=$5
stderr_log_path=$6

# load conda, path should be changed to the actual path on your cluster
# source /path/to/conda.sh
# conda activate /path/to/your_conda_env_with_sglang

NCCL_SOCKET_IFNAME=bond0 \
NCCL_IB_DISABLE=0 \
NCCL_DEBUG=INFO \
NCCL_IB_GID_INDEX=3 \
NCCL_NET_PLUGIN=none \
NCCL_IB_ECE_ENABLE=0 \
NVSHMEM_IB_ENABLE_IBGDA=0 \
NCCL_NVLS_ENABLE=0 \
NCCL_IB_TIMEOUT=22 \
NVSHMEM_IB_GID_INDEX=3 \
MC_TE_METRIC=True \
GLOO_SOCKET_IFNAME=bond0 \
MC_ENABLE_DEST_DEVICE_AFFINITY=True \
SGL_ENABLE_JIT_DEEPGEMM=True \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m sglang.launch_server \
    --model-path=$model_path \
    --host=0.0.0.0 \
    --port=$port \
    --tp-size=32 \
    --dp-size=32 \
    --ep-size=32 \
    --nnodes=4 \
    --dist-init-addr=$dist_init_addr \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static=0.7 \
    --mm-attention-backend=fa3 \
    --attention-backend=fa3 \
    --chunked-prefill-size=262144 \
    --moe-a2a-backend=deepep \
    --enable-two-batch-overlap \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-dense-tp-size=1 \
    --watchdog-timeout=1000000 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 32}' \
    --node-rank=$node_rank \
    1>$stdout_log_path \
    2>$stderr_log_path
