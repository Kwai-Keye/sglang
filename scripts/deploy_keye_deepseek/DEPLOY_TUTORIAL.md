# 4机H800x8卡部署SGLang Server简要教程

## 概述

本教程介绍如何在4台服务器（每台8张H800 GPU）上部署SGLang分布式推理服务。

**硬件配置：**
- 节点数：4台服务器
- 每节点GPU数：8张H800
- 总GPU数：32张

**并行配置：**
- Data Parallelism Attention (DP-Attn): 32
- Expert Parallelism (EP): 32

## 前置准备

1. **环境要求**
   - 所有节点已安装SGLang及相关依赖
   - 所有节点可访问模型文件（共享存储或每节点本地副本）
   - 节点间网络互通，支持InfiniBand（推荐）

2. **网络配置**
   - 确认所有节点使用bond0网络接口（或根据实际情况修改）

## 部署步骤

### 1. 准备部署脚本

确保 `run.sh` 脚本在所有节点上可执行：

```bash
chmod +x scripts/deploy_keye_deepseek/run.sh
```

### 2. 配置参数

准备以下参数信息：

- **model_path**: 模型路径（所有节点可访问）
- **dist_init_addr**: 主节点IP地址和端口（格式：IP:PORT）
- **port**: 每个节点上SGLang服务监听端口（建议不同节点使用不同端口）
- **node_rank**: 节点编号（0-3，主节点为0）
- **stdout_log_path**: 标准输出日志路径
- **stderr_log_path**: 标准错误日志路径

### 3. 启动服务

**在主节点（node_rank=0）启动：**

```bash
# Node 0 (主节点)
./scripts/deploy_keye_deepseek/run.sh \
    /path/to/model \
    192.168.1.100:29500 \
    30000 \
    0 \
    /path/to/logs/node0_stdout.log \
    /path/to/logs/node0_stderr.log
```

**在其他节点启动：**

```bash
# Node 1
./scripts/deploy_keye_deepseek/run.sh \
    /path/to/model \
    192.168.1.100:29500 \
    30001 \
    1 \
    /path/to/logs/node1_stdout.log \
    /path/to/logs/node1_stderr.log

# Node 2
./scripts/deploy_keye_deepseek/run.sh \
    /path/to/model \
    192.168.1.100:29500 \
    30002 \
    2 \
    /path/to/logs/node2_stdout.log \
    /path/to/logs/node2_stderr.log

# Node 3
./scripts/deploy_keye_deepseek/run.sh \
    /path/to/model \
    192.168.1.100:29500 \
    30003 \
    3 \
    /path/to/logs/node3_stdout.log \
    /path/to/logs/node3_stderr.log
```

**注意：**
- `dist_init_addr` 在所有节点上必须相同（指向主节点）
- `node_rank` 必须唯一（0-3）

### 4. 批量启动（可选）

可以使用SSH批量启动所有节点：

```bash
#!/bin/bash

# Configuration
MODEL_PATH="/path/to/model"
DIST_INIT_ADDR="192.168.1.100:29500"
BASE_PORT=30000
LOG_DIR="/path/to/logs"

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
```

## 关键配置说明

### NCCL环境变量

脚本中配置了以下关键NCCL参数：

- `NCCL_SOCKET_IFNAME=bond0`: 指定网络接口
- `NCCL_IB_DISABLE=0`: 启用InfiniBand
- `NCCL_IB_GID_INDEX=3`: InfiniBand GID索引
- `NCCL_DEBUG=INFO`: 调试信息级别


## 验证部署

### 1. 检查日志

查看各节点的日志文件，确认无错误：

```bash
# 检查主节点日志
tail -f /path/to/logs/node0_stdout.log
tail -f /path/to/logs/node0_stderr.log
```

### 2. 检查服务状态

服务启动成功后，可以通过API测试：

```bash
# 测试健康检查（在主节点端口）
curl http://192.168.1.100:30000/health
```

## 停止服务

在各节点上停止服务：

```bash
# 查找进程
ps aux | grep sglang.launch_server

# 停止进程
pkill -f sglang.launch_server
```

或使用批量停止脚本：

```bash
for node_ip in "${NODES[@]}"; do
    ssh $node_ip "pkill -f sglang.launch_server"
done
```

### 注意事项

- **Conda 环境**：在使用部署脚本前，请确保已正确加载包含sglang及其依赖库的python环境，以conda为例，命令如下：

```bash
# load conda path
source /path/to/your_conda.sh
conda activate /path/to/your_conda_env_with_sglang
```
