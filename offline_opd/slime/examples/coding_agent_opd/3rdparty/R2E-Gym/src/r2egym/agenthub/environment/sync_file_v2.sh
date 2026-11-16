#!/bin/bash
set -euo pipefail

# === 参数 ===
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash $0 <file_or_dir_to_broadcast> [source_rank]"
  exit 1
fi

ITEM_RAW="$1"
SRC_RANK="${2:-0}"            # 新增：源 rank（默认 0）
ITEM="${ITEM_RAW%/}"          # 去掉用户可能带的最后一个 '/'

# === 集群信息 ===
: "${PADDLE_TRAINERS:?PADDLE_TRAINERS is required (comma-separated hosts)}"
IFS=',' read -ra HOSTS <<< "${PADDLE_TRAINERS}"
RANK="${PADDLE_TRAINER_ID:-0}"

# 校验
[[ "$RANK" =~ ^[0-9]+$ ]] || { echo "[ERR] Invalid PADDLE_TRAINER_ID=$RANK"; exit 1; }
[[ "$SRC_RANK" =~ ^[0-9]+$ ]] || { echo "[ERR] Invalid source rank: $SRC_RANK"; exit 1; }
(( RANK >=0 && RANK < ${#HOSTS[@]} )) || { echo "[ERR] RANK out of range"; exit 1; }
(( SRC_RANK >=0 && SRC_RANK < ${#HOSTS[@]} )) || { echo "[ERR] source rank out of range"; exit 1; }

MASTER_ADDR="${HOSTS[0]}"         # 保留你的原逻辑中的 master 概念（展示用）
SOURCE_ADDR="${HOSTS[$SRC_RANK]}" # 由第二个参数指定真正的源节点
REMOTE_PWD="$(pwd)"
SRC_PATH="${REMOTE_PWD%/}/${ITEM}" # 远端统一绝对路径

# === 日志（极简） ===
echo "[Rank $RANK] Broadcasting: $ITEM  | SourceRank=$SRC_RANK Host=$SOURCE_ADDR  | Master=$MASTER_ADDR"

# === 执行逻辑（完全沿用你的判断范式） ===
if [[ "$RANK" -eq 0 && "$SRC_RANK" -ne 0 ]]; then
  # 仅提示：master 不一定是源
  : # 静默，避免日志噪声
fi

if [[ "$RANK" -eq "$SRC_RANK" ]]; then
  echo "[Rank $RANK] I am the source. No action needed."
  exit 0
fi

# === 同步（文件与目录一致处理：覆盖同名，目录合并，不删除多余） ===
# -q 静默  -a 归档  -H 硬链  -z 压缩  --inplace 原地覆盖，减少临时文件
RSYNC_OPTS=(-q -aHz --inplace)

echo "[Rank $RANK] Pulling from $SOURCE_ADDR ..."
rsync "${RSYNC_OPTS[@]}" "${SOURCE_ADDR}:${SRC_PATH}" "${REMOTE_PWD}/"
echo "[Rank $RANK] Done."
