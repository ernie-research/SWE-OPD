#!/bin/bash
set -ex

# ==============================================================================
# 8tr-8te-16ro : 8-GPU trainer (TP8 x CP1 x DP1) + 8-GPU teacher + 16-GPU rollout
#
# 相对基线 run-...-16tr-8te-8ro.sh 的**全部**差异（其余逐字保持一致，便于对照）：
#   1. ACTOR_NUM_NODES : NNODES-2 (=2 节点/16 卡)  ->  1 节点/8 卡
#   2. ROLLOUT_NUM_GPUS: 8                        ->  (NNODES-1-ACTOR_NUM_NODES)*8 = 16
#   3. TP_SIZE         : 4 (TP4 x DP4)            ->  8 (TP8 x DP1)
#   4. PERF_ARGS 追加   : --log-probs-chunk-size（只降 log_probs 前向峰值，见下）
#   5. 新增 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（抗变长打包的碎片）
# 目的：trainer 从 16 卡缩到 8 卡，把 8 张卡还给 rollout，压 perf/wait_time_ratio
# （基线实测 40 个 step 均值 0.795，train_time 中位 ~85s、train_wait 中位 ~310s）。
#
# ---------------- 为什么是 TP8 而不是 TP4xDP2 ----------------
# TP4xDP2 实测 OOM。DP 减半并不改变 micro-batch 峰值（`--use-dynamic-batch-size` +
# 固定 --max-tokens-per-gpu 决定峰值），变大的是**不随 TP/DP 切分的常驻数据**：
# actor.py:234-241 与 :278-291 把该 DP rank 的全部 tokens/loss_masks 和 OPD 支撑张量
# (student_support_ids [R,K] int64 + teacher_logprobs_on_student [R,K] fp32) 一次性搬上卡，
# 按 R≈20k / K=1024 约 250MB/样本 -> DP4 每卡 12 样本≈3GB，DP2≈6GB，DP1≈12GB，
# 而 DP4 稳态 free_GB 只剩 ~5.3。所以 DP1 会让常驻再涨 ~9GB，必须靠 TP8 省下的峰值补回来。
#
# TP8 合法性（脚本基线注释里"本模型最大 TP=4"是错的）：
#   * transformer_config.py:942  num_attention_heads 16 % 8 == 0                -> OK
#   * transformer_config.py:957  num_query_groups 需为 TP 的倍数**或约数**，4|8  -> OK
#     TP > num_query_groups 时 Megatron 走 KV 头复制：attention.py:1416 先 all-gather
#     mixed_qkv 再按 rank 切片，:1470-1480 再取本 rank 的 q 头。
#     ** 但 --attention-output-gate 在这条路上有 upstream bug**：attention.py:1491-1494 只
#     reshape gate，漏了 :1470-1480 那步 q 头切片，于是 gate 4 头 / core_attn_out 2 头，
#     _apply_output_gate 的 gate.view(*x.shape) 直接 RuntimeError。slime 侧已修：
#     slime/backends/megatron_utils/megatron_patch/megatron_output_gate_tp_patch.py
#     （TP <= num_query_groups 时该 patch 为 no-op，所以对基线脚本无影响）。
#   * GDN(linear_attention) 层：linear_num_key_heads=16 / value_heads=32，8 都能整除。
#
# TP8 能省多少（诚实版）：logits 是 vocab 切分的（model_provider.py:211 parallel_output=True，
# :55 在输出层就 .float()），每 rank 持 [1, T, V/TP] fp32：
#     T=131072, V=248320 -> TP4 = 32.5GB/份，TP8 = 16.3GB/份
# forward+backward 有多份（fp32 输出 / log_softmax / grad），这是最大的单一显存项。
# 但 32 层里 24 层是 linear_attention（config.json:19-52），其 GDN 模块用裸 nn.Linear
# (slime_plugins/models/qwen3_5.py:65-68,84)，**完全不做 TP 切分**，且 hf_attention.py:107
# 在 sequence-parallel 下会 all-gather 回全长序列 -> 这 24 层的权重/激活一分不省、通信还变多。
# 所以 TP8 是"只砍 logits + MLP + 8 层 full-attn"，不是全局减半。
#
# ---------------- 预期代价（跑之前先有心理准备）----------------
#   * DP1：48 条样本的 micro-batch 全部串行在一个 DP 组上（基线 DP4 是 4 路并行），
#     train_time 预计从 ~85s 涨到 250~350s 量级。若 rollout 侧降到 ~150s，
#     瓶颈会**从 rollout 翻转到 train**，wait_time_ratio 会掉到很低但总吞吐提升有限。
#     这正是本实验要量出来的数：对照基线看 perf/step_time 是否真的下降。
#   * DP1 让 distributed-optimizer 的 DP 切分失效（优化器状态已 CPU offload，影响有限）。
#   * 24 层 GDN 的 all-gather 随 TP 变大，step 变慢。
#
# ---------------- OOM 时的调参顺序（改一个跑一次）----------------
#   1. OPD_ID_TOPK=512 或 256   -> 常驻支撑张量线性下降（1024->256 即 /4，约省 9GB）
#   2. MAX_TOKENS_PER_GPU 下调  -> 只对"能被打包的"micro-batch 有效；单条超长样本仍独占
#                                  一个 micro-batch（dp_schedule.py:25-31），所以下界受
#                                  最长单序列约束（实测 response_len max ~58k）
#   3. LOG_PROBS_CHUNK_SIZE 再降 -> 只影响 log_probs 前向，不影响 train 峰值
#   * CP 这条路对当前 loss 是堵死的：loss.py:1340-1344 对 topk_{forward,reverse}_kl 直接
#     raise NotImplementedError（top-k 张量未做 CP 切分）。要用 CP 必须换回 token-level
#     OPD loss（OPD_ID_TOPK=0），那是建模决策，不在本脚本范围内。
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# ============ shared layer (see general_settings.sh) ============
# Provides, at source time: slime-venv PATH + RAY_LAUNCH, PDC topology (HOSTS / ALL_HOSTS /
# RANK / NNODES / MASTER_ADDR / POD_IP / ray ports / WAIT_WORKER_SECS), EXP_NAME + RUN_TAG +
# RUN_DIR with the per-rank awk log fan-out, NCCL/GLOO ifname + debug, cudnn/NVLink, the
# SWE+RLE sandbox env (incl. the RLE secret file), SLIME_HTTP_TIMING and
# SGLANG_TOP_LOGPROBS_IDS_B64, and the metrics switch/poller vars.
# Plus the helpers used below: preclean_ray, dump_engine_logs, metrics_poller,
# build_debug_args, setup_proxy, build_runtime_env_json.
# Everything layout- or teacher-specific (node roles, GPU split, teacher lifecycle,
# in-flight sizing, *_ARGS, ray bring-up) stays in THIS file.
source "${SCRIPT_DIR}/utils/general_settings.sh"

# ============ pick the DEDICATED teacher node ============
TEACHER_RANK="${TEACHER_RANK:-$(( NNODES - 1 ))}"
if [[ "${TEACHER_RANK}" -eq 0 ]]; then
   echo "TEACHER_RANK must not be 0 (rank0 is the ray head/submitter)." >&2
   exit 1
fi
TEACHER_IP="${TEACHER_IP:-${HOSTS[$TEACHER_RANK]}}"
if [[ "${RANK}" -eq "${TEACHER_RANK}" ]]; then IS_TEACHER=1; else IS_TEACHER=0; fi
echo "TEACHER_RANK=${TEACHER_RANK} TEACHER_IP=${TEACHER_IP} IS_TEACHER=${IS_TEACHER} (this RANK=${RANK})"

# ---- per-node ray GPU count ----
TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
RAY_GPUS="${RAY_GPUS:-0,1,2,3,4,5,6,7}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"

# ---- actor/rollout GPU split ----
# actor 只占 head 这一个节点（--placement-group-head-ip 把 head 的 bundle 排到最前，
# 见 placement_group.py:38-62 的 sort_key），剩下的非 teacher 节点全给 rollout。
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS_PER_NODE="${ROLLOUT_NUM_GPUS_PER_NODE:-8}"
# NNODES - 1(teacher) - ACTOR_NUM_NODES 个节点归 rollout；NNODES=4 -> 2 节点 = 16 卡。
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$(( (NNODES - 1 - ACTOR_NUM_NODES) * ROLLOUT_NUM_GPUS_PER_NODE ))}"
if [[ "${ROLLOUT_NUM_GPUS}" -le 0 ]]; then
   echo "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}: need NNODES > ACTOR_NUM_NODES + 1 (teacher)." >&2
   exit 1
fi
echo "GPU split: actor ${ACTOR_NUM_NODES} node(s) x ${ACTOR_NUM_GPUS_PER_NODE} GPU = $(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE )) (rank0 pinned to head ${MASTER_ADDR}) | rollout ${ROLLOUT_NUM_GPUS} GPU (${ROLLOUT_NUM_GPUS_PER_NODE}/node) | teacher on dedicated node ${TEACHER_IP} (out of ray)"

# ============ best-effort cleanup on EVERY node before ray start ============
# A stale ray head/GCS in /tmp/ray from a prior run makes `ray start` fail with a
# session-name mismatch. Does NOT touch a teacher that may already be serving (that reuse
# is handled in start_teacher).
preclean_ray

# Student (Qwen3.5-9B) megatron model args (in-repo, self-contained).
source "${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

WANDB_ARGS=(
   --use-wandb
   --disable-wandb-random-suffix
   --wandb-team swe_opd
   --wandb-project slime-OPD
   --wandb-group "${WANDB_GROUP:-"${EXP_NAME}-${RUN_TAG}"}"
   --wandb-id ${RUN_TAG}
   --wandb-mode online
   # stream this run's per-rank *.log files (RUN_DIR is on shared FS; the rank0 actor runs
   # on the head node where run.rank0.log lives) into the wandb run's Files tab, live.
   --wandb-log-dir "${RUN_DIR}"
)

# ============ context budget ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024*128))}"
# PER-TURN generation cap (-> --rollout-max-response-len). 0 = no per-turn cap: each
# turn gets the whole remaining context budget, exactly like the sibling
# package's eval harness (MAX_TOKENS=0, ../offline_opd eval/run_eval*.sh). Set a positive value (e.g. $((1024*16))) to cap a
# single turn; generate.py then treats a cut turn as "no valid tool call" and lets the
# model retry instead of dropping the trajectory.
MAX_GEN_LEN="${MAX_GEN_LEN:-0}"
# Student checkpoints. HF_CHECKPOINT is the ORIGINAL VLM dir (sglang rollout needs its
# image processor); load/save use the torch_dist megatron checkpoint. This is PURE OPD
# (reverse-KL toward the teacher server via opd_loss_function); no --ref-load needed.
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/path/to/Qwen3.5-9B}"
   --load "${LOAD_PATH:-/path/to/Qwen3.5-9B_torch_dist}"
   --save "${SAVE_PATH:-/path/to/Qwen3.5-9B_save_ckpt_scaleswe_topk512/}"
   --save-interval 40
)

# ============ teacher (external sglang server, Qwen3.6-27B) ============
# Dedicated node, ALL 8 GPUs. tp2 x dp4 = FOUR data-parallel replicas (each tp=2) behind a
# SINGLE port, with sglang's built-in DP load balancer. Each replica one-shot-prefills ONE
# request per forward (see --chunked-prefill-size -1 below), so the replica count IS the
# teacher's concurrency -- the right lever for the OPD logprob-fetch queue (reward_func
# fires one big topk_reverse_kl request per finished trajectory). Trading tp4->tp2 doubles
# per-request prefill latency and doubles the number of requests in flight; it only pays off
# if the client actually keeps >= dp requests posted (OPD_TEACHER_CONCURRENCY below).
# NB: 本布局把 rollout 从 8 卡加到 16 卡，完成轨迹的速率会上升 -> teacher 更容易成为下一个
# 瓶颈。观察 teacher 日志的 scheduler 队列与 GPU 利用率，必要时提 TEACHER_DP 或加节点。
TEACHER_PORT="${TEACHER_PORT:-13141}"
# Base of the PINNED torch-dist ports: replica i binds ${TEACHER_NCCL_PORT_BASE}+i (see the
# note above start_teacher for why this must not be left to sglang's random draw). Keep the
# whole band inside this cluster's ip_local_reserved_ports (2375,6676-18000,21270-21271,
# 23456,29500,30000-32768,34595,35000-61000) so the kernel never hands one of them to some
# other process's outbound connection.
TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-13150}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/Qwen3.6-27B}"
TEACHER_TP="${TEACHER_TP:-1}"
TEACHER_DP="${TEACHER_DP:-8}"
LAUNCH_TEACHER="${LAUNCH_TEACHER:-1}"
TEACHER_LOG="${RUN_DIR}/sglang_teacher.rank${RANK}.log"
TEACHER_BG_PID=""
# ---- boot robustness (see the port note above start_teacher) ----
#   TEACHER_LAUNCH_RETRIES     : how many times to (re)boot sglang before giving up
#   TEACHER_READY_TIMEOUT_SEC  : budget for ONE attempt to reach /health_generate. Measured
#                                on this layout: ~100s (27B x dp8 weight load + cuda graphs),
#                                so 600 is a 6x margin -- do not inflate it, it also bounds
#                                how long rank0 waits before declaring the run dead.
#   TEACHER_PORT_FREE_TIMEOUT_SEC : how long to wait for a reaped leftover to release
#                                   TEACHER_PORT before declaring the node unusable
TEACHER_LAUNCH_RETRIES="${TEACHER_LAUNCH_RETRIES:-3}"
TEACHER_READY_TIMEOUT_SEC="${TEACHER_READY_TIMEOUT_SEC:-600}"
TEACHER_PORT_FREE_TIMEOUT_SEC="${TEACHER_PORT_FREE_TIMEOUT_SEC:-60}"
# rank0 cannot see the teacher's log (RUN_DIR is node-local), so bound its wait: with the
# retry loop below a permanently broken teacher EXITS its pod, and an unbounded wait would
# hold the whole cluster idle forever. Default covers every attempt plus one.
TEACHER_WAIT_TIMEOUT_SEC="${TEACHER_WAIT_TIMEOUT_SEC:-$(( TEACHER_READY_TIMEOUT_SEC * (TEACHER_LAUNCH_RETRIES + 1) ))}"
# Every port this teacher binds explicitly: the HTTP port + one torch-dist port per replica.
TEACHER_PORTS=("${TEACHER_PORT}")
for (( _p = 0; _p < TEACHER_DP; _p++ )); do
   TEACHER_PORTS+=( "$(( TEACHER_NCCL_PORT_BASE + _p ))" )
done
echo "teacher ports: http=${TEACHER_PORT} torch-dist=${TEACHER_NCCL_PORT_BASE}..$(( TEACHER_NCCL_PORT_BASE + TEACHER_DP - 1 )) (all pinned, all inside ip_local_reserved_ports)"

teacher_healthy() {
   curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null 2>&1
}
# Anything LISTENing on port $1 on THIS node? (same probe cleanup.sh:149 uses)
port_listening() {
   ss -ltnH 2>/dev/null | grep -q ":$1 "
}
# Which of TEACHER_PORTS are busy on THIS node right now?
busy_teacher_ports() {
   local p out=""
   for p in "${TEACHER_PORTS[@]}"; do
      port_listening "${p}" && out+=" ${p}"
   done
   echo "${out# }"
}
# Reap a leftover teacher on THIS node and block until every pinned port is released.
# Only ever called on the dedicated teacher node, which never joins ray -> no rollout engine
# can be hit. The patterns are sglang's own process titles (`sglang::scheduler_DP0_TP0` etc,
# scheduler.py:3975) plus the launcher, bracketed like cleanup.sh:107 so they never match the
# cmdline of whatever shell is running this function. A bare `-f sglang` is also wrong: it
# matches general_settings' awk log dispatcher (its argv carries sglang_engine.rank*.log) and
# would kill this run's logging.
reap_stale_teacher() {
   local stale busy
   stale="$(pgrep -af "sglang[:]:|sglang[.]launch_server" || true)"
   busy="$(busy_teacher_ports)"
   if [[ -z "${stale}" && -z "${busy}" ]]; then
      echo "[teacher] no leftover sglang here, all ${#TEACHER_PORTS[@]} pinned ports free -> booting clean."
      return 0
   fi
   echo "[teacher] leftover sglang and/or busy ports (${busy:-none}) on this node -> reaping:"
   echo "${stale}"
   pkill -9 -f "sglang[:]:" || true
   pkill -9 -f "sglang[.]launch_server" || true
   local p
   for p in "${TEACHER_PORTS[@]}"; do
      fuser -k -9 "${p}/tcp" 2>/dev/null || true
   done
   rm -f /dev/shm/sglang* 2>/dev/null || true
   # Let the killed CUDA contexts actually release the 8 GPUs before the next boot claims
   # them (otherwise the retry can die on OOM instead of the port it was retrying for).
   sleep 5
   local waited=0
   while true; do
      busy="$(busy_teacher_ports)"
      [[ -z "${busy}" ]] && break
      if [[ "${waited}" -ge "${TEACHER_PORT_FREE_TIMEOUT_SEC}" ]]; then
         echo "[teacher] FATAL: still LISTENing after ${waited}s:${busy}, holder(s):" >&2
         ss -ltnp 2>/dev/null | grep -E ":(${busy// /|}) " >&2 || true
         echo "[teacher] not falling back to other ports: rank0 builds --rm-url from TEACHER_PORT on its OWN node, and the torch-dist band is pinned on purpose. Free them, or move TEACHER_PORT / TEACHER_NCCL_PORT_BASE (still inside ip_local_reserved_ports) for every node." >&2
         return 1
      fi
      sleep 3
      waited=$(( waited + 3 ))
   done
   echo "[teacher] all pinned ports released after ${waited}s."
}
# Poll one boot attempt. 0 = healthy; 1 = the sglang process died, or the attempt ran out of
# time. Never pkill on a health blip -- only a dead PID counts as failure.
wait_teacher_ready_local() {
   local waited=0
   while true; do
      if teacher_healthy; then
         echo "[teacher] healthy after ${waited}s (pid=${TEACHER_BG_PID})."
         return 0
      fi
      if ! kill -0 "${TEACHER_BG_PID}" 2>/dev/null; then
         echo "[teacher] sglang pid ${TEACHER_BG_PID} exited before becoming healthy (${waited}s)."
         return 1
      fi
      if [[ "${waited}" -ge "${TEACHER_READY_TIMEOUT_SEC}" ]]; then
         echo "[teacher] still not healthy after ${waited}s (TEACHER_READY_TIMEOUT_SEC)."
         return 1
      fi
      sleep 10
      waited=$(( waited + 10 ))
   done
}

# Launch the sglang teacher on THIS node's GPUs (idempotent: reuse if already up), retrying
# a failed boot. Called ONLY on the dedicated teacher node.
#
# ---- WHY the torch-dist ports are PINNED (measured on 10.0.0.11, 2026-07-29) ----
# Four consecutive boots died the same way, ~60s in, on ports 18025 / 18025 / 18033 / 18033:
#   torch.distributed.DistNetworkError ... port: 18025 ... EADDRINUSE
#   -> RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)
# Not TEACHER_PORT. With --dp-size 8 sglang draws ONE torch-dist port per replica from the
# kernel's ephemeral pool -- PortArgs.init_new -> get_free_port() = bind(port 0)
# (server_args.py:7812) -- the DP controller holds them, then closes ALL holder sockets
# (data_parallel_controller.py:246-278) BEFORE the 8 scheduler processes bind their TCPStore
# seconds later. On THIS node that window is a coin flip you lose:
#   * ip_local_port_range = 10000-61000 but ip_local_reserved_ports carves out
#     6676-18000 / 30000-32768 / 35000-61000 / ... -> the drawable pool is a narrow band
#     starting at 18001;
#   * `ss -tan | grep -c ':180xx'` = 628 sockets: the node's own agents (kubelet 10256,
#     node-local-dns 169.254.20.10:8080, the 1191/8177 mesh) churn outbound connections
#     exactly there;
#   * bind(0) is NOT uniformly random -- repeated draws on this node return the same low
#     ports first (measured: 18025, 18029, 18033, then 18061, 18063, 18065 in EVERY round).
# So sglang keeps handing its schedulers the hottest ports on the box, and an agent takes
# them during the gap. Retrying re-draws from the same hot zone -> retries do not help
# (that is exactly what the 3-attempt loop below proved).
# FIX: --nccl-port ${TEACHER_NCCL_PORT_BASE}, i.e. replica i binds BASE+i inside the
# RESERVED band (13150+, cf. 6676-18000) which the kernel never auto-assigns to anyone.
# Upstream sglang gives every replica the SAME nccl_port when it is set explicitly (which
# cannot work: one TCPStore each), so the vendored checkout adds the +dp_rank offset in
# data_parallel_controller.launch_dp_schedulers -- keep sglang in sync across nodes
# (`bash dispatch.sh sync /path/to/workspace/env_run/sglang`) or the teacher
# node boots the unpatched code and every replica collides on BASE.
# The retry loop below is kept for genuinely transient boot failures; with pinned ports the
# port class of failure is gone, and a truly occupied port now fails fast (bind_port in the
# controller) instead of a minute later inside a scheduler.
start_teacher() {
   if teacher_healthy; then
      echo "Teacher already serving on ${TEACHER_IP}:${TEACHER_PORT}; reusing it."
      return 0
   fi
   local attempt
   for (( attempt = 1; attempt <= TEACHER_LAUNCH_RETRIES; attempt++ )); do
      reap_stale_teacher || exit 1
      echo "[teacher] boot attempt ${attempt}/${TEACHER_LAUNCH_RETRIES} ..."
      launch_teacher_process
      if wait_teacher_ready_local; then
         return 0
      fi
      echo "[teacher] attempt ${attempt} failed; tail of ${TEACHER_LOG}:"
      tail -n 40 "${TEACHER_LOG}" 2>/dev/null || true
      if grep -q "EADDRINUSE" "${TEACHER_LOG}" 2>/dev/null; then
         echo "[teacher] -> random torch-dist port collision; the next attempt re-draws all ${TEACHER_DP} ports."
      fi
      # The parent may already be gone while its scheduler children still hold GPUs; the next
      # reap_stale_teacher clears them. Keep this attempt's log for the post-mortem.
      kill -9 "${TEACHER_BG_PID}" 2>/dev/null || true
      TEACHER_BG_PID=""
      mv "${TEACHER_LOG}" "${TEACHER_LOG}.attempt${attempt}" 2>/dev/null || true
   done
   echo "[teacher] FATAL: teacher did not come up in ${TEACHER_LAUNCH_RETRIES} attempts (logs: ${TEACHER_LOG}.attempt*)." >&2
   exit 1
}

# ============ teacher metrics CLI flags (native sglang args, gated by ENABLE_TEACHER_METRICS) ============
TEACHER_METRICS_ARGS=()
if [[ "${ENABLE_TEACHER_METRICS}" -eq 1 ]]; then
   TEACHER_METRICS_ARGS=(
      --enable-metrics
      --enable-metrics-for-all-schedulers
      --enable-cache-report
      --decode-log-interval "${SGLANG_DECODE_LOG_INTERVAL:-100}"
   )
fi

# ONE sglang boot; sets TEACHER_BG_PID. Only start_teacher calls this.
launch_teacher_process() {
   # --chunked-prefill-size -1 (one-shot prefill) is REQUIRED for topk_reverse_kl:
   # reward_func sends a per-position token_ids_logprob matrix for the whole response
   # region. With chunked prefill sglang raises logprob_start_len to the last chunk's
   # boundary, so pruned_len shrinks while the id matrix stays full-length ->
   # get_token_ids_logprobs_raw asserts and the teacher SIGQUITs.
   # SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK: chunks the input-logprob logits over scored
   # POSITIONS (peak logits memory = chunk_size x vocab x 4B instead of response_len x
   # vocab x 4B) -- a different axis from --chunked-prefill-size, which stays at -1.
   # SGLANG_OPD_TOPK_OVERLAP: teacher appends |TopK_teacher ∩ student ids| as one extra
   # column per scored position, which post_process_rewards divides by K and the loss logs
   # as train/opd_topk_overlap. Computed teacher-side because it needs the teacher's own
   # full-vocab top-k, which is never shipped back; costs one topk over the scored
   # positions, so unset it to drop both the cost and the metric.
   SGLANG_OPD_TOPK_OVERLAP=1 \
   SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1 SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=$((1024*8)) \
   CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" python3 -m sglang.launch_server \
      --model-path "${TEACHER_MODEL}" \
      --host 0.0.0.0 \
      --port ${TEACHER_PORT} \
      --tp ${TEACHER_TP} \
      --dp-size ${TEACHER_DP} \
      --nccl-port ${TEACHER_NCCL_PORT_BASE} \
      --trust-remote-code \
      --tool-call-parser qwen3_coder \
      --attention-backend "${TEACHER_ATTN_BACKEND:-flashinfer}" \
      --chunked-prefill-size -1 \
      --disable-radix-cache \
      --max-prefill-tokens $((1024*256)) \
      --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.8}" \
      "${TEACHER_METRICS_ARGS[@]}" \
      > "${TEACHER_LOG}" 2>&1 &
   TEACHER_BG_PID=$!
   echo "Starting teacher server (${TEACHER_MODEL}) on ${POD_IP} GPUs ${TEACHER_GPUS} (tp=${TEACHER_TP} dp=${TEACHER_DP}) pid=${TEACHER_BG_PID}, log=${TEACHER_LOG}"
}
# Block until the teacher is healthy (used by rank0 before submit; the teacher is on a
# remote dedicated node here). Bounded by TEACHER_WAIT_TIMEOUT_SEC -> returns 1 instead of
# pinning the whole cluster on a teacher that will never come up.
wait_teacher() {
   local waited=0
   until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
      if [[ "${waited}" -ge "${TEACHER_WAIT_TIMEOUT_SEC}" ]]; then
         echo "FATAL: teacher at ${TEACHER_IP}:${TEACHER_PORT} not healthy after ${waited}s; inspect ${RUN_DIR}/sglang_teacher.rank${TEACHER_RANK}.log* ON ${TEACHER_IP} (RUN_DIR is node-local)." >&2
         return 1
      fi
      echo "Waiting for teacher server at ${TEACHER_IP}:${TEACHER_PORT}... (${waited}s/${TEACHER_WAIT_TIMEOUT_SEC}s)"
      [[ "${IS_TEACHER}" -eq 1 ]] && tail -n 5 "${TEACHER_LOG}" 2>/dev/null || true
      sleep 5
      waited=$(( waited + 5 ))
   done
   curl -s http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info || true
   echo "Teacher server is up at ${TEACHER_IP}:${TEACHER_PORT}."
}

# ROLLOUT: student drives the sweagent tool loop; fully-async rollout function.
# Hoisted to named vars (single source of truth): the in-flight / RLE-container ceiling
# below is derived from ROLLOUT_BATCH_SIZE x N_SAMPLES_PER_PROMPT.
# 与基线保持一致（48/24/2）：本实验只改 trainer/rollout 的卡数划分，batch 宽度不动，
# 否则 perf/step_time 的对照就不干净。DP=1 时 global_batch_size 无整除约束。
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-2} #NUM_STEPS_PER_ROLLOUT 个 step 全部跑完之后同步一次
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}" # prompt/group 为单位
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}" 
# sample 为单位，一次 optimizer step 吃多少条 sample
# GLOBAL_BATCH_SIZE= \
# "${GLOBAL_BATCH_SIZE:-$(((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT) / NUM_STEPS_PER_ROLLOUT))}"
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.coding_agent_opd.generate.generate
   --prompt-data "${PROMPT_DATA:-${SLIME_DIR}/data/scale_swe_python_sample5000_train_ccr.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-120}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --num-steps-per-rollout ${NUM_STEPS_PER_ROLLOUT}
   --log-multi-turn
   # --global-batch-size 64
   --balance-data
)
# REWARD: reward_func POSTs the token sequence to the sglang teacher (--rm-url,
# return_logprob=True); coding_agent_opd's post-process promotes metadata["swe_reward"] to
# the scalar reward and attaches the teacher logprobs the OPD loss consumes.
RM_ARGS=(
   --custom-rm-path examples.coding_agent_opd.opd.reward_func
   --custom-reward-post-process-path examples.coding_agent_opd.opd.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# fully-async rollout raises on eval, so no --eval-interval here.
EVAL_ARGS=(
)

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}}"
# --log-probs-chunk-size：把 log_probs/entropy 的 [T, V/TP] 按位置分块（loss.py:519），
# 峰值从 T x Vlocal 降到 chunk x Vlocal。**只作用于 log_probs 这条非 loss 通道**
# （get_log_probs_and_entropy 在 loss.py:505 断言 non_loss_data），
# 不会降低 train forward/backward 的峰值 —— 不要把它当 OOM 的解药。
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
PERF_ARGS=(
   # TP8 x CP1 x DP1 on 8 GPUs。TP 把 vocab 切 8 份，OPD 的 [T, V/TP] fp32 logits
   # 从 TP4 的 32.5GB/份降到 16.3GB/份（V=248320, T=131072），这是本布局能站住的关键。
   # 注意 24/32 层是 linear_attention，其 GDN 模块不做 TP 切分（见文件头说明）。
   --tensor-model-parallel-size "${TP_SIZE:-8}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   # CP 对 topk_{forward,reverse}_kl 不可用（loss.py:1340 直接 raise NotImplementedError）。
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-loss-function
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --recompute-granularity full # 重算整层，而非局部
   --recompute-method uniform # 均匀地对每一层做重算
   --recompute-num-layers 1 # 每个重算单元包含 1 层
   --use-dynamic-batch-size # 动态批量大小 Packing样本
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

# OPD: reverse-KL toward the teacher server's token logprobs. --opd-type sglang fetches
# them via the reward hook above (cross-arch teacher OK: shared vocab).
# ---- OPD loss variant (--opd-loss-type) ----
#   * topk_forward_kl : forward-KL over the TEACHER's per-position top-k support.
#   * topk_reverse_kl : reverse-KL over the STUDENT's per-position top-k support (student
#                       top-k ids queried on the teacher via per-position token_ids_logprob).
# Set OPD_LOSS_TYPE + OPD_ID_TOPK to switch; this script defaults to topk_reverse_kl with
# K=1024 (slime's own default is the token-level loss), and OPD_ID_TOPK=0 falls back to it.
# ==> OOM 时的第一个旋钮就是这里：K 直接决定 [R, K] 支撑张量的常驻显存
#     （student_support_ids int64 + teacher_logprobs_on_student fp32，
#      actor.py:278-291 一次性搬上卡，且不随 TP/DP 切分）。DP=1 下 48 条样本约 12GB，
#      K=1024->256 可省约 9GB。改 K 会改变 reverse-KL 的支撑集宽度，属训练语义变更。
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-topk_reverse_kl}"
OPD_ID_TOPK="${OPD_ID_TOPK:-512}"
OPD_ARGS=(
   --opd-type sglang
   --loss-type custom_loss
   --custom-loss-function-path slime.backends.megatron_utils.loss.opd_loss_function
   --opd-loss-type "${OPD_LOSS_TYPE}"
   --opd-id-topk "${OPD_ID_TOPK}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ============ async in-flight pool sizing ============
# Only THREE values leave this block: ROLLOUT_GPUS_PER_ENGINE, NUM_ROLLOUT_ENGINES and
# SGLANG_SERVER_CONCURRENCY. Everything else is arithmetic on ONE designed quantity:
#   designed pool = INFLIGHT_STEPS x rollout_batch_size groups
#                 = that x n_samples_per_prompt SAMPLES   <- the unit slime actually counts
# slime's generate semaphore is per SAMPLE (sglang_rollout.generate_and_rm acquires it once
# per sample; fully_async_rollout._get_global_worker divides it back by n_samples_per_prompt
# to size the group pool) and --sglang-server-concurrency is PER ENGINE, so the knob is
# ceil(samples / engines). That rounding up makes slime admit slightly MORE than designed,
# and the admitted sample count IS the RLE container ceiling (one pod per sample, alive for
# the whole multi-turn trajectory -- this is what can cross the session TTL), so the echo
# reports the admitted numbers rather than the requested ones.
# Anchor to the DESIGNED batch, NOT the engine count, so the pool doesn't balloon when
# engines are added. INFLIGHT_STEPS must be >=2 so rollout(N+1) overlaps train(N).
# 本布局引擎数从 4 变 8（16 卡 / 2 卡每引擎），设计池保持 192 samples 不变 -> 每引擎并发
# 从 48 降到 24，单样本解码更快、KV 压力更小，而 RLE 容器上限与基线相同（对照干净）。
# RLE 有余量，想进一步喂满 16 卡可以提 INFLIGHT_STEPS，但那会同时抬高沙箱并发。
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
INFLIGHT_STEPS="${INFLIGHT_STEPS:-4}"

NUM_ROLLOUT_ENGINES=$(( ROLLOUT_NUM_GPUS / ROLLOUT_GPUS_PER_ENGINE ))
INFLIGHT_SAMPLES=$(( INFLIGHT_STEPS * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))

SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-$(( (INFLIGHT_SAMPLES + NUM_ROLLOUT_ENGINES - 1) / NUM_ROLLOUT_ENGINES ))}"

ADMITTED_SAMPLES=$(( SGLANG_SERVER_CONCURRENCY * NUM_ROLLOUT_ENGINES ))
echo "in-flight sizing: ${INFLIGHT_STEPS} steps x ${ROLLOUT_BATCH_SIZE} groups x ${N_SAMPLES_PER_PROMPT} samples = ${INFLIGHT_SAMPLES} samples designed | ${NUM_ROLLOUT_ENGINES} engines -> sglang_server_concurrency=${SGLANG_SERVER_CONCURRENCY} -> admits ${ADMITTED_SAMPLES} samples = $(( ADMITTED_SAMPLES / N_SAMPLES_PER_PROMPT )) groups = RLE container ceiling"

# Fallback port list for the KV-metrics poller (kv_dashboard.sh prefers `ss` discovery):
# slime's engines start at 15000 and step by rollout_num_gpus_per_engine, so the list is a
# function of THIS layout's rollout split -- derive it instead of hardcoding.
if [[ -z "${METRICS_ENGINE_PORTS:-}" ]]; then
   METRICS_ENGINE_PORTS=""
   for (( i = 0; i < NUM_ROLLOUT_ENGINES; i++ )); do
      METRICS_ENGINE_PORTS+="$(( 15000 + i * ROLLOUT_GPUS_PER_ENGINE )) "
   done
fi
echo "metrics engine ports (poller fallback): ${METRICS_ENGINE_PORTS}"

# ============ sglang profiling / metrics CLI flags for rollout (gated by ENABLE_SGLANG_METRICS) ============
SGLANG_METRICS_ARGS=()
if [[ "${ENABLE_SGLANG_METRICS}" -eq 1 ]]; then
   SGLANG_METRICS_ARGS=(
      --sglang-enable-metrics
      --sglang-enable-metrics-for-all-schedulers
      --sglang-enable-cache-report
      --sglang-decode-log-interval "${SGLANG_DECODE_LOG_INTERVAL:-100}"
   )
fi
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.85
   --tool-call-parser qwen3_coder
   --use-distributed-post
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-attention-backend "${SGLANG_ATTN_BACKEND:-flashinfer}"
   "${SGLANG_METRICS_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============ optional: dump trajectories + train data for post-hoc analysis ============
# Policy lives in general_settings.sh: DUMP_DETAILS=1 (default) writes under
# ${TRAJ_ROOT}/${EXP_NAME}_${RUN_TAG}, rollout-data only unless DUMP_TRAIN_DATA=1, capped by
# DUMP_DETAILS_NUM_ROLLOUTS. Override any of those before this call.
build_debug_args

# ============ allocator: 抗变长打包造成的碎片 ============
# 稳态 reserved 251.9GB / allocated ~21GB（基线 rank0 日志）说明 caching allocator 抓着大量
# 空闲块。expandable_segments 让 CUDA 分配段可增长，变长 micro-batch 下能显著降低
# "有空闲但装不下"的 OOM。仅当外部没设过时才设，方便对照实验。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============ teacher (OPD rm_url) HTTP timeouts + concurrency ============
#   OPD_TEACHER_TIMEOUT_SEC          -> aiohttp `total` (connected-but-slow response)
#   OPD_TEACHER_CONNECT_TIMEOUT_SEC  -> aiohttp `sock_connect` (fires when the teacher is
#     saturated and stops accepting connections).
#   OPD_TEACHER_CONCURRENCY          -> concurrent teacher scoring calls. slime's rollout
#     semaphore wraps only the GENERATE call, so uncapped EVERY finished trajectory POSTs
#     its ~R*K*4B base64 blob at once. Measured effect of that flood: teacher scheduler
#     queue 0, teacher GPUs ~0%, client disconnects in the teacher log -- it starves the
#     teacher instead of feeding it.
#     Default = 2x the DP replica count: one request prefilling per replica, one queued
#     behind it.
export OPD_TEACHER_TIMEOUT_SEC="${OPD_TEACHER_TIMEOUT_SEC:-7200}"
export OPD_TEACHER_CONNECT_TIMEOUT_SEC="${OPD_TEACHER_CONNECT_TIMEOUT_SEC:-360}"
export OPD_TEACHER_CONCURRENCY="${OPD_TEACHER_CONCURRENCY:-$((TEACHER_DP * 2))}"

# ============ proxy: github fetch via proxy, in-cluster traffic direct ============
# no_proxy covers the teacher, every trainer pod and the RLE gateways (see general_settings.sh).
setup_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# unset http_proxy
# unset https_proxy
# export no_proxy="localdomain,localhost.localdomain,localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,*.lan,.example.com,.internal.example.com,example.com,internal.example.com,baidubce.com,10.0.0.17"
# export NO_PROXY=${NO_PROXY}

cd "${SLIME_DIR}"

# ============ runtime env propagated to ray workers ============
# Key list + PATH/PYTHONPATH/TP_SOCKET_IFNAME live in general_settings.sh; it forwards the
# OPD_TEACHER_* values exported above and PYTORCH_CUDA_ALLOC_CONF exported here.
build_runtime_env_json

# ============ per-node branch on PADDLE_TRAINER_ID ============
if [[ "${IS_TEACHER}" -eq 1 ]]; then
   # -------- DEDICATED TEACHER node: serve the 27B, never join ray --------
   if [[ "${LAUNCH_TEACHER}" -ne 1 ]]; then
      echo "[teacher] LAUNCH_TEACHER=0 on the teacher node -> nothing to do; exiting."
      exit 0
   fi
   start_teacher   # probes/frees TEACHER_PORT, boots, retries a lost port race; exits on failure
   wait_teacher || exit 1   # already healthy here: this only logs /get_model_info
   echo "[teacher] serving on ${TEACHER_IP}:${TEACHER_PORT} (tp=${TEACHER_TP} dp=${TEACHER_DP}); blocking on the sglang process until it exits or the pod is reclaimed."
   # Block on the sglang PROCESS, NOT on /health_generate: a long teacher forward makes
   # /health return 503 while the server is merely BUSY, and pkill-ing on that 503 kills a
   # healthy teacher. `wait` on the PID only returns when sglang actually exits.
   if [[ -n "${TEACHER_BG_PID}" ]]; then
      # `set -e` would abort on a nonzero `wait`, so capture the rc explicitly --
      # otherwise the echo below can only ever print rc=0.
      TEACHER_RC=0
      wait "${TEACHER_BG_PID}" || TEACHER_RC=$?
      echo "[teacher] sglang process exited (rc=${TEACHER_RC}); nothing to kill."
   else
      # start_teacher reused an already-serving teacher (no PID owned here): just idle so
      # the pod stays alive. Do NOT pkill on a transient health blip.
      while true; do sleep 3600; done
   fi
   exit 0
elif [[ "${RANK}" -eq 0 ]]; then
   # -------- HEAD (rank0): ray head + submit (THE actor node) --------
   if [[ "${LAUNCH_TEACHER}" -eq 1 ]]; then
      # remote teacher on the dedicated node must be live before submit (rm-url). Bounded:
      # if it never comes up, fail here instead of holding the cluster idle forever.
      wait_teacher || exit 1
      sleep 5
   else
      echo "LAUNCH_TEACHER=0: expecting an external teacher at ${TEACHER_IP}:${TEACHER_PORT}."
   fi

   # ray head on this node's 8 GPUs. --placement-group-head-ip=${MASTER_ADDR} (passed to
   # train_async.py) pins these bundles to the FRONT so the WHOLE actor (8 GPU, TP8) lands
   # on rank0 and every other node's bundles go to rollout.
   CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --head --node-ip-address "${POD_IP}" \
      --port "${RAY_HEAD_PORT}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

   echo "[head] waiting ${WAIT_WORKER_SECS}s for ray nodes to join (expect $(( NNODES - 1 )) ray nodes)..."
   sleep "${WAIT_WORKER_SECS}"
   ray status || true
   # ============ submit (driver = train_async.py, disaggregated) ============
   METRICS_POLLER_PID=""
   if [[ "${ENABLE_SGLANG_METRICS}" -eq 1 && "${METRICS_POLL_INTERVAL}" -gt 0 ]]; then
      echo "[head] starting sglang metrics poller (interval=${METRICS_POLL_INTERVAL}s) -> ${METRICS_LOG}"
      metrics_poller 2>&1 | tee -a "${METRICS_LOG}" &
      METRICS_POLLER_PID=$!
   fi

   SUBMIT_RC=0
   ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${SLIME_VENV}/bin/python3" -u train_async.py \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
      --num-gpus-per-node "${ROLLOUT_NUM_GPUS_PER_NODE}" \
      --placement-group-head-ip "${MASTER_ADDR}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${OPD_ARGS[@]}" \
      "${WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      "${DEBUG_ARGS[@]}" \
      "${RM_ARGS[@]}" || SUBMIT_RC=$?

   if [[ "${SUBMIT_RC}" -ne 0 ]]; then
      echo "[head] ray job submit failed (rc=${SUBMIT_RC}); dumping engine logs for post-mortem."
      dump_engine_logs
   fi

   # ============ cleanup after training (head only) ============
   [[ -n "${METRICS_POLLER_PID}" ]] && kill "${METRICS_POLLER_PID}" 2>/dev/null || true
   pkill -9 sglang || true  > /dev/null 2>&1
   sleep 3
   ray stop --force || true > /dev/null 2>&1
   exit "${SUBMIT_RC}"
else
   # -------- WORKER (rollout only in this layout): join the head, block so the pod stays alive --------
   echo "[worker ${RANK}] joining ray head at ${MASTER_ADDR}:${RAY_HEAD_PORT} (num-gpus=${RAY_NUM_GPUS_PER_NODE})"
   until CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
                   --node-ip-address "${POD_IP}" \
                   --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats --block; do
      echo "[worker ${RANK}] ray start failed, retry in 5s..."
      ray stop --force || true  > /dev/null 2>&1
      sleep 5
   done
   echo "[worker ${RANK}] ray cluster stopped; dumping engine logs for post-mortem."
   dump_engine_logs
fi
