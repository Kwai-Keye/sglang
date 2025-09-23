#!/bin/bash


# Configuration
MODEL_PATH="/path/to/model"
DIST_INIT_ADDR="192.168.1.100:29500"
BASE_PORT=30000
LOG_DIR="/path/to/logs"
mkdir -p $LOG_DIR

# Node IPs
NODES=("192.168.1.100" "192.168.1.101" "192.168.1.102" "192.168.1.103")

# Launch on each node
for i in "${!NODES[@]}"; do
    node_ip="${NODES[$i]}"
    node_rank=$i
    port=$((BASE_PORT + i))

    echo "Launching on node $node_rank (IP: $node_ip)"

    ssh $node_ip "cd /path/to/sglang && \
        ./scripts/deploy_keye_deepseek/run.sh \
        $MODEL_PATH \
        $DIST_INIT_ADDR \
        $port \
        $node_rank \
        $LOG_DIR/node${node_rank}_stdout.log \
        $LOG_DIR/node${node_rank}_stderr.log" &
done

wait
echo "All nodes launched"
