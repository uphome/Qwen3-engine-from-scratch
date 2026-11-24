#!/usr/bin/env bash
# 下载 Qwen3-0.6B 权重（默认到 ./models/Qwen3-0.6B）
#
# 用法:
#   bash scripts/download_model.sh                 # 默认 Qwen/Qwen3-0.6B -> models/Qwen3-0.6B
#   bash scripts/download_model.sh models/Qwen3-1.7B
#   MODEL_ID=Qwen/Qwen3-1.7B bash scripts/download_model.sh

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
DEST="${1:-models/Qwen3-0.6B}"

mkdir -p "$DEST"

python - "$MODEL_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(model_id, local_dir=dest)
print(f"[download] {model_id} -> {path}")
PY

echo "[download] 完成。运行示例:"
echo "  python main.py --model $DEST --temperature 0"
