"""
双足压力 + 膝盖拉伸传感器的实时步态/姿态识别器。
===================================================================================
整体思路是分层判断，核心是严格的 4095 判据（膝盖传感器满量程 = 腿伸直）。

  1) 先做自适应预处理（DualFootAdaptiveBank），再走滑动窗口。

  2) 第一层：膝盖 ACTIVE / INACTIVE 门控（严格 4095 规则）。

     判断逻辑：
       * 只要有任意一侧膝盖的原始 ADC 没顶到 4095（也就是
         min(raw_knee_l, raw_knee_r) < KNEE_RAW_STRAIGHT_TH），
         那这一帧就算 ACTIVE 候选。
       * 只有两侧膝盖都贴着 4095 跑，才算 INACTIVE 候选。

     为了避免站着不动时因为噪声来回抖，我加了防抖：
       用滑窗累计 sustain / release 两个计数器（KNEE_GATE_SUSTAIN_SAMPLES /
       KNEE_GATE_RELEASE_SAMPLES），再配一个最短保持时间
       （KNEE_GATE_MIN_HOLD_S），保证分支状态不会一下被小抖动翻过去。

  3) 第二层：MOTION / STATIC 细分。

     * ACTIVE 分支里：
        - MOTION  → 上楼 / 下楼（有周期性的 4095 尖峰 + 真实迈步事件）
        - STATIC  → 正常坐 / 翘二郎腿（整段都到不了 4095）

     * INACTIVE 分支要求更严（这是我自己定的规矩，防止站着抖被误判成走路）：
        - 必须在最近窗口里看到至少 WALK_ENTER_MIN_STEPS 次"真正抬脚"的事件
          （用双足接触检测，不只看压力大小），同时还要有同样数量的膝盖摆动
          到 ≥ 4095 轨的记录，才允许从 STATIC 切到 MOTION（前走 / 后走）。
        - 站着时身体微小的重心晃动会被压住，不会误触发。

  4) 分支 RF：根据前两层的结果，只调四个独立模型中的一个：
        rf_active_motion.joblib   → STAIRS_UP / STAIRS_DOWN
        rf_active_static.joblib   → SITTING_NORMAL / SITTING_CROSSLEGGED
        rf_inactive_motion.joblib → WALKING_FORWARD / WALKING_BACKWARD
        rf_inactive_static.joblib → STANDING_UPRIGHT / LEFT_LEAN / RIGHT_LEAN

步态签名的那些标量作为辅助特征喂进去，布局和训练时保持一致，
具体看 ml_branch_models.build_auxiliary_vector 里的写法。

SIT_TO_STAND 是纯规则判的，不走 RF。

依赖：numpy、collections.deque、time（不用 scipy，不想引太多依赖）
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from ml_activity_features import WINDOW_SIZE as ML_WINDOW_SIZE
from ml_branch_models import (
    BranchRFEnsemble,
    build_auxiliary_vector,
    build_full_feature_vector,
    BRANCH_TO_FILE,
)
from adaptive_preprocessing import CHANNEL_NAMES_DUAL, DualFootAdaptiveBank

# ═══════════════════════════════════════════════════════════════════════════
#  这一堆是可调参数，后面调阈值主要就是改这里
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_MAX = 4095.0

# 要和单片机固件的出帧频率保持一致。目前硬件是每 100ms 发一帧双足数据
# （用 CSV 时间戳差值验证过），也就是 10 Hz。
# 如果改了这个值，ml_activity_features.SAMPLE_HZ 也要跟着改，而且四个 RF 模型都要重训。
SAMPLE_HZ = 10

# EMA 平滑系数，范围 (0, 1]，越小越平滑但越滞后
EMA_ALPHA = 0.25

# ── 脚跟迈步检测（旧方案，留着兜底） ─────────────────────────────────

# 脚跟信号再过一次 EMA，只用在步检测里
HEEL_STEP_EMA_ALPHA = 0.35

# Schmitt 触发器的初始上下阈值，累计到 3 步以上会根据最近的峰谷自动重新估算
STEP_INIT_LOW = 0.25
STEP_INIT_HIGH = 0.45

# 每只脚的迈步冷却时间（秒），0.30s 对应单脚最多 200 步/分钟，正常人肯定够用
STEP_COOLDOWN_S = 0.30
STEP_COOLDOWN_SAMPLES = max(4, int(round(SAMPLE_HZ * STEP_COOLDOWN_S)))

# 自适应阈值用最近多少个峰 / 谷做平均
ADAPTIVE_HISTORY = 10

# 峰谷差小于这个就认为信号太弱、不够稳定，不拿来更新阈值
ADAPTIVE_MIN_SWING = 0.04

# 上下阈值取峰谷之间的这个比例位置
ADAPTIVE_LOW_FRAC = 0.30
ADAPTIVE_HIGH_FRAC = 0.60

# 追峰追谷用的迟滞（按当前峰谷差的比例算）
PEAK_TROUGH_HYST = 0.08

# ── 脚底接触检测（主用的双足迈步方案） ────────────────────────────────

# 单脚三区压力之和低于这个，就当抬脚了
FOOT_OFF_GROUND_TH = 0.10

# 单脚三区压力之和高于这个，就当落地了
FOOT_ON_GROUND_TH = 0.20

# 同一只脚两次有效迈步之间的最小间隔（秒），防抖
STEP_MIN_GAP_S = 0.30

# 连续多少帧低于抬脚阈值才判定为"抬起来了"
FOOT_OFF_MIN_SAMPLES = 3

# 连续多少帧高于落地阈值才判定为"踩下去了"
FOOT_ON_MIN_SAMPLES = 3

# ── 方向判断 ────────────────────────────────────────────────────────

# 每踩一步之后，后续观察窗口长度（秒），用来对比脚跟和脚尖先到达峰值
WINDOW_SECONDS = 0.6

# 最近几步做多数投票来决定方向
VOTE_K_STEPS = 3

# ── 第二层：MOTION / STATIC 细分 ────────────────────────────────────

# 整体运动幅度太小的话，就别进 MOTION 分支了（可能只是抖一抖）
MOTION_MIN_AMPLITUDE = 0.10

# 连续几帧都同时满足"有步伐 + 幅度够"，才确认进入 MOTION
MOTION_CONFIRM_FRAMES = 4

# 进 MOTION 之后最少待够这么久，才允许切回 STATIC，避免来回抖
MOTION_MIN_HOLD_S = 1.0

# 在 MOTION 里连续这么多帧都没证据了，才允许切回 STATIC
LAYER2_STATIC_CONFIRM_FRAMES = 6

# ── 走路进入的强保护（只在 INACTIVE 分支用） ─────────────────────────
# 这是为了防止"站着身体微抖"被错判成走路。
# INACTIVE 分支下从 STATIC 切到 MOTION，额外要求：
#   (a) 最近窗口里有至少 WALK_ENTER_MIN_STEPS 次真实抬脚事件；并且
#   (b) 最近 WALK_EVIDENCE_WINDOW_S 秒内，膝盖至少有 WALK_ENTER_MIN_STEPS 次
#       伸到接近 4095 轨的证据（也就是摆腿确实到位了）。

# 进入 WALKING 前需要的最少真实迈步数
WALK_ENTER_MIN_STEPS = 2
# 累计迈步 + 膝盖伸直证据的滚动窗口（秒）
WALK_EVIDENCE_WINDOW_S = 2.0
# 膝盖原始 ADC 达到这个值才算"这一下摆腿伸直了"
WALK_KNEE_EXTEND_RAIL_TH = 4080.0

# ── 状态机 ──────────────────────────────────────────────────────────

# 一个状态进入后至少保持这么久才允许切换（秒）
STATE_MIN_DURATION_S = 1.0

# 最后一次迈步之后过了这么久没新步，就退出 WALKING / STAIRS
WALK_TIMEOUT_S = 2.0

# ── 坐姿检测 ────────────────────────────────────────────────────────

# 脚底总压力小于这个，判为坐着（脚上基本没负重）
SIT_PSUM_TH = 0.08

# 坐着时脚底压力标准差超过这个，倾向于翘腿
SIT_CROSSLEG_STD_TH = 0.04

# 坐着时膝盖偏离基线超过这个，也倾向于翘腿
SIT_CROSSLEG_KNEE_TH = 0.15

# ── 第一层膝盖门控（ACTIVE / INACTIVE 切换） ────────────────────────
# 严格 4095 规则（顶上模块 docstring 有写）：
#   • 瞬时 ACTIVE  ← min(raw_knee_l, raw_knee_r) <  KNEE_RAW_STRAIGHT_TH
#     （至少一条腿是弯的，没顶到 4095）
#   • 瞬时 INACTIVE ← min(raw_knee_l, raw_knee_r) >= KNEE_RAW_STRAIGHT_TH
#     （两条腿都顶在 4095 附近）

# 膝盖 ADC 高于这个就当"伸直"（4095 是满量程）。
# 往 4095 靠 → ACTIVE 更灵敏，一点点弯就触发；
# 往下压（比如 3000）→ 会忽略走路时小幅的膝盖弯，只对上下楼 / 坐反应。
KNEE_RAW_STRAIGHT_TH = 3500.0

# 连续几帧满足瞬时 ACTIVE 规则才真正切到 ACTIVE。
# 大了更抗单帧误判，代价是切换变慢。
KNEE_GATE_SUSTAIN_SAMPLES = max(3, int(round(SAMPLE_HZ * 0.40)))
# 连续几帧不满足 ACTIVE 规则才掉回 INACTIVE。
# 大了在两步楼梯之间短暂伸直时也不会掉出 ACTIVE。
KNEE_GATE_RELEASE_SAMPLES = max(3, int(round(SAMPLE_HZ * 0.35)))
# 每次第一层切换后，新分支至少锁这么久（秒），防止站坐临界来回闪。
KNEE_GATE_MIN_HOLD_S = 0.6

# ── 站立检测 ────────────────────────────────────────────────────────

# 没迈步但 p_sum >= 这个阈值，就是站着
STAND_PSUM_TH = 0.15

# 左右偏重的阈值，按 lr_ratio 判断
LEAN_LR_TH = 0.12

# ── 坐起（sit-to-stand）────────────────────────────────────────────

# 坐起触发 / 确认阈值（都在压力域，也就是归一化之后的）
STS_TRIGGER_TH = 0.15
STS_CONFIRM_TH = 0.25
STS_CONFIRM_S = 0.8

# 膝盖压力要比坐着的基线高出这么多才算"有要站起来的趋势"
STS_KNEE_DELTA_TH = 0.08

# STS_TREND_SAMPLES 这段时间内 p_sum 最小到最大要涨这么多才算有上升趋势
STS_MIN_PSUM_RISE = 0.02

# 看 p_sum 趋势的窗口长度（帧数，在这个窗口里找 min/max）
STS_TREND_SAMPLES = 10

# 单足流走 8 通道自适应模型时，缺的那只脚原始 ADC 用这个值填
_SINGLE_FOOT_PAD_RAW = 2048.0


def _snapshot_to_adaptive_debug_dict(s) -> dict:
    return {
        "raw": round(float(s.raw), 2),
        "baseline_raw": round(float(s.baseline_raw), 2),
        "baseline_removed": round(float(s.baseline_removed), 4),
        "relative_pressure_ratio": round(float(s.relative_pressure_ratio), 4),
        "adaptive_zscore": round(float(s.adaptive_zscore), 4),
        "current_state": s.stable_state,
        "confidence": round(float(s.confidence), 4),
        "dynamic_min_raw": round(float(s.dynamic_min_raw), 2),
        "dynamic_max_raw": round(float(s.dynamic_max_raw), 2),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  第二层：MOTION / STATIC
# ═══════════════════════════════════════════════════════════════════════════


class _Layer2MotionStatic:
    """判 MOTION / STATIC，靠迈步证据 + 幅度迟滞。

    如果第一层是 INACTIVE（站 ↔ 走），在 WALK_EVIDENCE_WINDOW_S
    这段滚动窗口里必须同时有 WALK_ENTER_MIN_STEPS 次真实抬脚和同样多的
    膝盖伸直见证，才允许进 MOTION。这么做就是为了把站着身体晃动导致的
    假迈步压住。
    """

    def __init__(self) -> None:
        self._sub = "STATIC_BRANCH"
        self._up_cnt = 0
        self._down_cnt = 0
        self._motion_enter_t = -999.0
        self._last_evidence_t = -999.0
        self._reason = "init"
        self._step_events: deque[float] = deque()
        self._knee_extend_events: deque[float] = deque()

    def _trim(self, dq: deque[float], t: float) -> None:
        cutoff = t - WALK_EVIDENCE_WINDOW_S
        while dq and dq[0] < cutoff:
            dq.popleft()

    def update(
        self,
        raw_step: bool,
        recent_step: bool,
        amp: float,
        t: float,
        *,
        layer1_branch: str = "INACTIVE_BRANCH",
        knee_extended_now: bool = False,
    ) -> tuple[str, str]:
        if raw_step:
            self._step_events.append(t)
        if knee_extended_now:
            self._knee_extend_events.append(t)
        self._trim(self._step_events, t)
        self._trim(self._knee_extend_events, t)

        evidence = (raw_step or recent_step) and amp >= MOTION_MIN_AMPLITUDE
        if evidence:
            self._last_evidence_t = t

        # 只有第一层是 INACTIVE 时才加这个保护：必须同时看到真实迈步 + 膝盖摆动
        walk_guard_ok = True
        walk_guard_reason = ""
        if layer1_branch == "INACTIVE_BRANCH":
            need = WALK_ENTER_MIN_STEPS
            n_steps = len(self._step_events)
            n_knee = len(self._knee_extend_events)
            walk_guard_ok = (n_steps >= need) and (n_knee >= need)
            if not walk_guard_ok:
                walk_guard_reason = (
                    f"walk_guard_waiting(steps={n_steps}/{need},"
                    f"knee_ext={n_knee}/{need})"
                )

        if self._sub == "STATIC_BRANCH":
            if evidence and walk_guard_ok:
                self._up_cnt += 1
                self._down_cnt = 0
                if self._up_cnt >= MOTION_CONFIRM_FRAMES:
                    self._sub = "MOTION_BRANCH"
                    self._motion_enter_t = t
                    self._up_cnt = 0
                    self._reason = "motion_confirmed"
            else:
                self._up_cnt = 0
                self._reason = walk_guard_reason or "static_no_evidence"
            return self._sub, self._reason

        # 进到这里就是已经在 MOTION_BRANCH 里了
        if evidence:
            self._down_cnt = 0
            self._reason = "motion_sustained"
            return self._sub, self._reason

        self._down_cnt += 1
        self._reason = "motion_cooling"
        hold_ok = (t - self._motion_enter_t) >= MOTION_MIN_HOLD_S
        if self._down_cnt >= LAYER2_STATIC_CONFIRM_FRAMES and hold_ok:
            self._sub = "STATIC_BRANCH"
            self._down_cnt = 0
            self._reason = "static_hold"
        elif (t - self._last_evidence_t) > WALK_TIMEOUT_S and hold_ok:
            self._sub = "STATIC_BRANCH"
            self._down_cnt = 0
            self._reason = "static_timeout"
        return self._sub, self._reason


# ═══════════════════════════════════════════════════════════════════════════
#  小工具
# ═══════════════════════════════════════════════════════════════════════════

def raw_to_pressure(raw: float) -> float:
    return float(np.clip((SENSOR_MAX - raw) / SENSOR_MAX, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  EMA 滤波
# ═══════════════════════════════════════════════════════════════════════════

class _EMAFilter:
    def __init__(self, alpha: float = EMA_ALPHA):
        self._a = alpha
        self._v: float | None = None

    def update(self, x: float) -> float:
        if self._v is None:
            self._v = x
        else:
            self._v += self._a * (x - self._v)
        return self._v

    @property
    def value(self) -> float:
        return 0.0 if self._v is None else self._v


# ═══════════════════════════════════════════════════════════════════════════
#  自适应迈步检测器（按脚算，Schmitt 阈值自动调）
#  旧版思路，基于脚跟信号，现在主要当 fallback 和调试用
# ═══════════════════════════════════════════════════════════════════════════

class _AdaptiveStepDetector:
    """单脚的迈步检测器，Schmitt 阈值会自己调。

    记一下最近的脚跟峰和谷，攒够 3 步之后用最近
    ADAPTIVE_HISTORY 个峰 / 谷的平均值重新算上下阈值。
    这样不管鞋垫里用的是哪种电阻都能自适应。
    """

    def __init__(self) -> None:
        self._low = STEP_INIT_LOW
        self._high = STEP_INIT_HIGH
        self._armed = True
        self._cooldown = 0
        self._prev = 0.0

        # 追峰追谷
        self._recent_peaks: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._recent_troughs: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._tracking_val = 0.0
        self._phase = "seek_peak"   # "seek_peak" | "seek_trough"

    # ── 对外接口 ─────────────────────────────────────────────────────
    def update(self, heel_smooth: float) -> bool:
        self._track_peaks_troughs(heel_smooth)

        step = False
        if self._cooldown > 0:
            self._cooldown -= 1
        else:
            if (
                self._armed
                and self._prev < self._high
                and heel_smooth >= self._high
            ):
                step = True
                self._armed = False
                self._cooldown = STEP_COOLDOWN_SAMPLES
        if heel_smooth <= self._low:
            self._armed = True
        self._prev = heel_smooth
        return step

    @property
    def thresholds(self) -> tuple[float, float]:
        return self._low, self._high

    # ── 内部 ─────────────────────────────────────────────────────────
    def _track_peaks_troughs(self, v: float) -> None:
        hyst = max(PEAK_TROUGH_HYST, (self._high - self._low) * 0.25)
        if self._phase == "seek_peak":
            if v > self._tracking_val:
                self._tracking_val = v
            elif v < self._tracking_val - hyst:
                self._recent_peaks.append(self._tracking_val)
                self._tracking_val = v
                self._phase = "seek_trough"
                self._recalc()
        else:
            if v < self._tracking_val:
                self._tracking_val = v
            elif v > self._tracking_val + hyst:
                self._recent_troughs.append(self._tracking_val)
                self._tracking_val = v
                self._phase = "seek_peak"
                self._recalc()

    def _recalc(self) -> None:
        if len(self._recent_peaks) < 3 or len(self._recent_troughs) < 3:
            return
        avg_pk = float(np.mean(list(self._recent_peaks)))
        avg_tr = float(np.mean(list(self._recent_troughs)))
        swing = avg_pk - avg_tr
        if swing < ADAPTIVE_MIN_SWING:
            return
        self._low  = avg_tr + ADAPTIVE_LOW_FRAC * swing
        self._high = avg_tr + ADAPTIVE_HIGH_FRAC * swing


class _StepDetectorV2(_AdaptiveStepDetector):
    """老代码里有地方 import 这个名字，留个别名兼容一下。"""
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  脚底接触迈步检测（按脚走"抬起 → 落地"一次事件）
# ═══════════════════════════════════════════════════════════════════════════

class _FootContactStepDetector:
    """按脚底三区压力之和（toe + forefoot + heel）来数脚步。

    状态转移：
      ON_GROUND  --（连续 N 帧压力 < FOOT_OFF_GROUND_TH）--> LIFTED
      LIFTED     --（连续 N 帧压力 > FOOT_ON_GROUND_TH ）--> ON_GROUND，记一步

    用连续帧计数 + 两步最小间隔来防抖，避免噪声触发假步。
    """

    def __init__(self) -> None:
        self._phase = "ON_GROUND"   # "ON_GROUND" | "LIFTED"
        self._off_count = 0
        self._on_count = 0
        self._last_step_t = -999.0

    def update(self, load: float, t: float) -> bool:
        """喂一帧数据，返回 True 就表示这一帧踩出了一个有效步（重新落地）。"""
        step = False
        if self._phase == "ON_GROUND":
            if load < FOOT_OFF_GROUND_TH:
                self._off_count += 1
                if self._off_count >= FOOT_OFF_MIN_SAMPLES:
                    self._phase = "LIFTED"
                    self._on_count = 0
            else:
                self._off_count = 0
        elif self._phase == "LIFTED":
            if load > FOOT_ON_GROUND_TH:
                self._on_count += 1
                if self._on_count >= FOOT_ON_MIN_SAMPLES:
                    self._phase = "ON_GROUND"
                    self._off_count = 0
                    if (t - self._last_step_t) >= STEP_MIN_GAP_S:
                        step = True
                        self._last_step_t = t
            else:
                self._on_count = 0
        return step

    @property
    def phase(self) -> str:
        return self._phase


# ═══════════════════════════════════════════════════════════════════════════
#  方向判断（比较脚尖和脚跟哪个先到峰值）
# ═══════════════════════════════════════════════════════════════════════════

class _DirectionDetector:
    """脚跟先到峰 → 向前走；脚尖先到峰 → 向后走"""

    def __init__(self):
        self._collecting = False
        self._t0 = 0.0
        self._buf: list[tuple[float, float, float]] = []

    def on_step(self, t: float):
        self._collecting = True
        self._t0 = t
        self._buf.clear()

    def feed(self, t: float, toe_p: float, heel_p: float) -> str | None:
        if not self._collecting:
            return None
        self._buf.append((t, toe_p, heel_p))
        if (t - self._t0) < WINDOW_SECONDS:
            return None
        self._collecting = False
        if not self._buf:
            return "unknown"
        t_toe = max(self._buf, key=lambda r: r[1])[0]
        t_heel = max(self._buf, key=lambda r: r[2])[0]
        if abs(t_heel - t_toe) < 0.02:
            return "unknown"
        return "forward" if t_heel < t_toe else "backward"


# ═══════════════════════════════════════════════════════════════════════════
#  方向投票（最近 K 步少数服从多数）
# ═══════════════════════════════════════════════════════════════════════════

class _DirectionVoting:
    def __init__(self, k: int = VOTE_K_STEPS):
        self._buf: deque[str] = deque(maxlen=k)

    def push(self, d: str):
        if d != "unknown":
            self._buf.append(d)

    @property
    def result(self) -> str:
        if not self._buf:
            return "unknown"
        fwd = sum(1 for d in self._buf if d == "forward")
        bwd = len(self._buf) - fwd
        if fwd > bwd:
            return "forward"
        if bwd > fwd:
            return "backward"
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  坐起检测（sit-to-stand）
# ═══════════════════════════════════════════════════════════════════════════

class _SitToStandDetector:
    """坐起这个动作我没用模型，纯靠规则：
    脚底总压力 + 膝盖相对基线的变化 + p_sum 短时上升趋势。
    """

    def __init__(self) -> None:
        self._phase = "idle"
        self._trigger_t = 0.0
        self._confirm_t = 0.0
        self.last_duration: float | None = None
        self._psum_ring: deque[float] = deque(maxlen=max(3, STS_TREND_SAMPLES))
        self._last_p_sum = 0.0

    def update(
        self,
        p_sum: float,
        p_knee: float,
        sm_state: str,
        knee_baseline: float | None,
        t: float,
    ) -> tuple[bool, bool]:
        """
        返回 (force_sit_to_stand, post_complete_standing)。
        如果第二个是 True，说明 last_duration 刚被赋值，外面应该在这一帧
        直接给一个站着的候选标签。
        """
        self._last_p_sum = p_sum
        self._psum_ring.append(p_sum)
        trend_ok = True
        if len(self._psum_ring) >= 3:
            trend_ok = (max(self._psum_ring) - min(self._psum_ring)) >= STS_MIN_PSUM_RISE

        is_sitting = sm_state.startswith("SITTING")
        knee_ok = True
        if knee_baseline is not None:
            knee_ok = (p_knee - knee_baseline) >= STS_KNEE_DELTA_TH

        if self._phase == "idle":
            if (
                is_sitting
                and p_sum >= STS_TRIGGER_TH
                and knee_ok
                and trend_ok
            ):
                self._phase = "triggered"
                self._trigger_t = t
            return (False, False)

        if self._phase == "triggered":
            if p_sum < STS_TRIGGER_TH * 0.5:
                self._phase = "idle"
                return (False, False)
            if p_sum >= STS_CONFIRM_TH:
                self._phase = "confirming"
                self._confirm_t = t
            return (True, False)

        if self._phase == "confirming":
            if p_sum < STS_CONFIRM_TH:
                self._phase = "triggered"
                return (True, False)
            if (t - self._confirm_t) >= STS_CONFIRM_S:
                self.last_duration = t - self._trigger_t
                self._phase = "idle"
                self._psum_ring.clear()
                return (False, True)
            return (True, False)

        self._phase = "idle"
        return (False, False)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trigger_level(self) -> float:
        th = STS_TRIGGER_TH if STS_TRIGGER_TH > 1e-9 else 1e-9
        return float(self._last_p_sum / th)

    @property
    def confirm_level(self) -> float:
        th = STS_CONFIRM_TH if STS_CONFIRM_TH > 1e-9 else 1e-9
        return float(self._last_p_sum / th)


# ═══════════════════════════════════════════════════════════════════════════
#  状态机（带防抖和最短保持时间）
# ═══════════════════════════════════════════════════════════════════════════

ALL_STATES = {
    "WALKING_FORWARD", "WALKING_BACKWARD",
    "STAIRS_UP", "STAIRS_DOWN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
    "SIT_TO_STAND",
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
    "UNKNOWN",
}

# RF 训练用的标签集合（ALL_STATES 的子集），把 SIT_TO_STAND 去掉，因为那是规则判的
RF_TRAINING_LABELS = ALL_STATES - {"SIT_TO_STAND"}


class _StateMachine:
    def __init__(self):
        self.state = "UNKNOWN"
        self._entered_t = 0.0

    def propose(self, candidate: str, t: float, *, immediate: bool = False) -> str:
        if candidate == self.state:
            return self.state
        if immediate or (t - self._entered_t) >= STATE_MIN_DURATION_S:
            self.state = candidate
            self._entered_t = t
        return self.state


# ═══════════════════════════════════════════════════════════════════════════
#  主类：OnlineRecognizer
# ═══════════════════════════════════════════════════════════════════════════

class OnlineRecognizer:
    """每来一帧就调一次 update() 或 update_bilateral()。

    返回 dict，字段有：
      state, step_event, walk_dir, counters, sts_last_duration_s, debug
    """

    _STEP_STATES = {"WALKING_FORWARD", "WALKING_BACKWARD", "STAIRS_UP", "STAIRS_DOWN"}

    def __init__(
        self,
        calibration: "object | str | None" = "auto",
    ):
        """
        参数说明
        ----------
        calibration : PersonalCalibration | str（路径）| "auto" | None
            * 直接传 PersonalCalibration 对象 — 就用它。
            * 传字符串 — 当成 JSON 路径，可能是离线自动标定的结果，也可能是
              界面上那个两步在线标定向导生成的。
            * "auto"（默认）— 尝试从当前工作目录读 personal_calibration.json；
              找不到就当 None，安静降级。
            * None — 完全关掉个人标定（老行为，EWMA bank 直接吃原始 ADC）。

            第一层的膝盖门控（严格 4095 规则，KNEE_RAW_STRAIGHT_TH）永远读的是
            标定之前的原始 ADC，所以"单腿 raw ≠ 4095 → ACTIVE"这条规则不会被
            这个参数影响。
        """
        # 个人标定，可选
        self._calibration = self._resolve_calibration(calibration)
        # 左脚四通道 EMA
        self._f_toe  = _EMAFilter(EMA_ALPHA)
        self._f_ff   = _EMAFilter(EMA_ALPHA)
        self._f_heel = _EMAFilter(EMA_ALPHA)
        self._f_knee = _EMAFilter(EMA_ALPHA)

        # 右脚四通道 EMA（只在 update_bilateral 里用）
        self._f_toe_r  = _EMAFilter(EMA_ALPHA)
        self._f_ff_r   = _EMAFilter(EMA_ALPHA)
        self._f_heel_r = _EMAFilter(EMA_ALPHA)
        self._f_knee_r = _EMAFilter(EMA_ALPHA)

        # 脚跟版迈步检测（旧方案，兜底 + 调试）
        self._heel_step_l = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._heel_step_r = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._step_det_l  = _AdaptiveStepDetector()
        self._step_det_r  = _AdaptiveStepDetector()
        self._step_det    = _AdaptiveStepDetector()   # 单脚 fallback

        # 脚底接触迈步检测（双足主用）
        self._contact_det_l      = _FootContactStepDetector()
        self._contact_det_r      = _FootContactStepDetector()
        self._contact_det_single = _FootContactStepDetector()

        # 方向
        self._dir_det  = _DirectionDetector()
        self._dir_vote = _DirectionVoting(VOTE_K_STEPS)
        self._last_valid_walk_dir: str | None = None

        # 坐起
        self._sts_det = _SitToStandDetector()

        # 状态机
        self._sm = _StateMachine()

        # 四分支 RF 模型集合（默认从工作目录下加载 *.joblib）
        self._branch_models = BranchRFEnsemble()
        self._layer2 = _Layer2MotionStatic()
        self._last_adaptive_snaps: list | None = None

        self._counters = {
            "forward_steps":  0,
            "backward_steps": 0,
            "up_steps":       0,
            "down_steps":     0,
            "total_steps":    0,
        }
        self._last_step_t = 0.0

        # 翘腿检测用的膝盖基线，用最开始 N 帧学一个值
        self._knee_baseline: float | None = None
        self._knee_init_buf: list[float] = []

        # 在线自适应预处理，8 通道（单脚路径会把另一只脚补上）。
        # 如果标定里带了全局统计量，就把每个通道都种到那套 baseline /
        # 压力范围 / 均值 / 方差上，保证推理看到的数值分布和训练时一致。
        # 如果标定里只有 [min_raw, max_raw]（老版本或者 identity 标定），
        # 就退化成纯 EWMA 跑。
        _calib = self._calibration
        _seeds = None
        if _calib is not None and getattr(_calib, "has_global_stats", False):
            _seeds = _calib.to_channel_seeds()
        self._adapt_bank = DualFootAdaptiveBank(seeds=_seeds)

        # 喂给 ML 滑窗的压力 / 特征历史：
        # 老版是 4 或 8 个 float（压力），adaptive_v2 是 24 个 float（8 通道 × [br, ratio, z]）
        self._p_hist: deque[np.ndarray] = deque(
            maxlen=max(ML_WINDOW_SIZE, int(SAMPLE_HZ * 2)),
        )
        self._motion_amp_hist: deque[float] = deque(maxlen=max(8, int(SAMPLE_HZ * 0.6)))
        self._zone_contact_state = {"toe": False, "forefoot": False, "heel": False}
        self._zone_event_hist: deque[tuple[str, str, float]] = deque(maxlen=24)

        # 第一层膝盖门控（严格 4095 + 防抖 + 最短保持）
        self._layer1_branch = "INACTIVE_BRANCH"
        self._knee_gate_sustain_cnt = 0
        self._knee_gate_release_cnt = 0
        self._layer1_last_switch_t = -999.0

    def _knee_gate_instant_active(self, min_raw_knee: float) -> bool:
        """严格 4095 规则：只要有一只膝盖没顶到 4095，瞬时判 ACTIVE。

        min_raw_knee = min(raw_knee_l, raw_knee_r)；单脚路径会把缺的那侧
        填成 4095，规则就自动退化成只看一只膝盖。
        raw ADC >= KNEE_RAW_STRAIGHT_TH 就当是"直的"（差不多到 4095）。
        """
        return float(min_raw_knee) < KNEE_RAW_STRAIGHT_TH

    def _update_layer1_knee_gate(
        self, min_raw_knee: float, t: float,
    ) -> tuple[str, dict[str, object]]:
        """带防抖的第一层门控：sustain + release 计数 + 最短保持时间。"""
        cond = self._knee_gate_instant_active(min_raw_knee)
        phase = "inactive"
        hold_ok = (t - self._layer1_last_switch_t) >= KNEE_GATE_MIN_HOLD_S

        if self._layer1_branch == "INACTIVE_BRANCH":
            if cond:
                self._knee_gate_sustain_cnt += 1
                self._knee_gate_release_cnt = 0
                if self._knee_gate_sustain_cnt >= KNEE_GATE_SUSTAIN_SAMPLES and hold_ok:
                    self._layer1_branch = "ACTIVE_BRANCH"
                    self._knee_gate_sustain_cnt = 0
                    self._layer1_last_switch_t = t
                    phase = "active"
                else:
                    phase = "arming" if hold_ok else "locked_inactive"
            else:
                self._knee_gate_sustain_cnt = 0
                self._knee_gate_release_cnt = 0
                phase = "inactive"
        else:
            if not cond:
                self._knee_gate_release_cnt += 1
                self._knee_gate_sustain_cnt = 0
                if self._knee_gate_release_cnt >= KNEE_GATE_RELEASE_SAMPLES and hold_ok:
                    self._layer1_branch = "INACTIVE_BRANCH"
                    self._knee_gate_release_cnt = 0
                    self._layer1_last_switch_t = t
                    phase = "inactive"
                else:
                    phase = "releasing" if hold_ok else "locked_active"
            else:
                self._knee_gate_release_cnt = 0
                phase = "active"

        info: dict[str, object] = {
            "knee_gate_phase": phase,
            "knee_gate_min_raw": round(float(min_raw_knee), 1),
            "knee_gate_straight_th": float(KNEE_RAW_STRAIGHT_TH),
            "knee_gate_sustain_cnt": int(self._knee_gate_sustain_cnt),
            "knee_gate_sustain_need": int(KNEE_GATE_SUSTAIN_SAMPLES),
            "knee_gate_release_cnt": int(self._knee_gate_release_cnt),
            "knee_gate_release_need": int(KNEE_GATE_RELEASE_SAMPLES),
            "knee_gate_min_hold_s": float(KNEE_GATE_MIN_HOLD_S),
        }
        return self._layer1_branch, info

    def _compute_min_knee_raw(
        self,
        raw_knee_l: float,
        raw_knee_r: float,
        *,
        single_foot: bool = False,
    ) -> float:
        """第一层用的指标：两侧膝盖 raw ADC 的最小值，也就是弯得最厉害那一侧。"""
        if single_foot:
            return float(raw_knee_l)
        return float(min(raw_knee_l, raw_knee_r))

    @staticmethod
    def _resolve_calibration(arg):
        """把构造函数里的 calibration= 参数统一解析成 PersonalCalibration
        或者 None。离线和在线两套路径都走这里，保证行为一致。
        """
        if arg is None:
            return None
        try:
            import personal_calibration as _pc
        except Exception as exc:       # pragma: no cover — 兜底，理论上不会走到
            print(f"[OnlineRecognizer] personal_calibration unavailable: {exc}")
            return None
        if isinstance(arg, _pc.PersonalCalibration):
            return arg
        if arg == "auto":
            calib = _pc.load_default_calibration((".",))
            if calib is not None:
                print(f"[OnlineRecognizer] loaded calibration "
                      f"(source={calib.source!r}, subject={calib.subject!r})")
            return calib
        if isinstance(arg, str):
            return _pc.PersonalCalibration.load_json(arg)
        raise TypeError(f"Unsupported calibration argument type: {type(arg)!r}")

    def _calibrate_raw8(self, raw8: np.ndarray) -> np.ndarray:
        """如果加载了个人标定，就按每个通道的 [min, max] 线性映射到 [0, 4095]。
        注意第一层膝盖门控不走这里，它读的是未标定的 raw，保证严格 4095
        规则在标定之后还成立。
        """
        if self._calibration is None:
            return raw8
        return np.asarray(
            self._calibration.normalize_to_adc(raw8), dtype=np.float64,
        )

    def _branch_key_from_layers(self, active: bool, motion_sub: str) -> str:
        if active:
            return (
                "active_motion"
                if motion_sub == "MOTION_BRANCH"
                else "active_static"
            )
        return (
            "inactive_motion"
            if motion_sub == "MOTION_BRANCH"
            else "inactive_static"
        )

    def _hierarchical_rf_predict(
        self,
        branch_key: str,
        sign: dict[str, object],
        left_load: float,
        right_load: float,
        lr_ratio: float,
        p_sum: float,
    ) -> tuple[str, float, str, str]:
        fname = BRANCH_TO_FILE.get(branch_key, "")
        b = self._branch_models.bundle(branch_key)
        if not b.available:
            return "UNKNOWN", 0.0, "model_missing", fname
        aux = build_auxiliary_vector(sign, left_load, right_load, lr_ratio, p_sum)
        win = np.array(list(self._p_hist)[-ML_WINDOW_SIZE:])
        if win.ndim != 2 or win.shape[1] != 24:
            return "UNKNOWN", 0.0, "window_not_adaptive_v2", fname
        feat = build_full_feature_vector(win, aux)
        if feat is None:
            return "UNKNOWN", 0.0, "feature_error", fname
        lab, pr, rj = b.predict(feat)
        return lab, float(pr), rj, fname

    # ── 对外 API ──────────────────────────────────────────────────────

    def update(
        self,
        raw_toe: float,
        raw_forefoot: float,
        raw_heel: float,
        raw_knee: float,
        t: float | None = None,
    ) -> dict:
        """单脚更新（4 通道）。"""
        if t is None:
            t = time.monotonic()

        adaptive_dbg: dict[str, dict] = {}
        raw8 = np.array(
            [
                raw_toe,
                raw_forefoot,
                raw_heel,
                raw_knee,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
            ],
            dtype=np.float64,
        )
        raw8_calib = self._calibrate_raw8(raw8)
        _flat24, snaps = self._adapt_bank.update(raw8_calib)
        self._last_adaptive_snaps = list(snaps)
        p_toe = self._f_toe.update(snaps[0].relative_pressure_ratio)
        p_ff = self._f_ff.update(snaps[1].relative_pressure_ratio)
        p_heel = self._f_heel.update(snaps[2].relative_pressure_ratio)
        p_knee = self._f_knee.update(snaps[3].relative_pressure_ratio)
        p_sum = p_toe + p_ff + p_heel
        self._p_hist.append(_flat24)
        for nm, sn in zip(CHANNEL_NAMES_DUAL, snaps):
            adaptive_dbg[nm] = _snapshot_to_adaptive_debug_dict(sn)

        if self._knee_baseline is None:
            self._knee_init_buf.append(p_knee)
            if len(self._knee_init_buf) >= 30:
                self._knee_baseline = float(np.mean(self._knee_init_buf))
        min_raw_knee = self._compute_min_knee_raw(raw_knee, 0.0, single_foot=True)
        layer1, knee_gate_info = self._update_layer1_knee_gate(min_raw_knee, t)
        knee_extended_now = float(raw_knee) >= WALK_KNEE_EXTEND_RAIL_TH
        self._update_zone_sequence(p_toe, p_ff, p_heel, t)

        contact_step = self._contact_det_single.update(p_sum, t)
        step_source: str | None = "single_contact" if contact_step else None
        heel_smooth = self._heel_step_l.update(p_heel)
        heel_step = self._step_det.update(heel_smooth)
        raw_step = contact_step
        if not raw_step and heel_step:
            raw_step = True
            step_source = "heel_fallback"

        if raw_step:
            self._dir_det.on_step(t)
            self._last_step_t = t

        per_step_dir = self._dir_det.feed(t, p_toe, p_heel)
        if per_step_dir is not None:
            self._dir_vote.push(per_step_dir)
        walk_dir = self._dir_vote.result
        if walk_dir in ("forward", "backward"):
            self._last_valid_walk_dir = walk_dir

        recent_step = (t - self._last_step_t) < WALK_TIMEOUT_S
        amp = self._estimate_motion_amplitude(p_toe, p_ff, p_heel, p_knee)
        layer2, layer2_reason = self._layer2.update(
            raw_step,
            recent_step,
            amp,
            t,
            layer1_branch=layer1,
            knee_extended_now=knee_extended_now,
        )
        sign = self._gait_signature_from_pressures(
            p_toe, p_ff, p_heel, p_knee,
        )
        br_key = self._branch_key_from_layers(
            layer1 == "ACTIVE_BRANCH", layer2,
        )
        cand, rf_p, rf_rej, rf_name = self._hierarchical_rf_predict(
            br_key,
            sign,
            p_sum,
            0.0,
            0.0,
            p_sum,
        )

        force_sts, post_sts = self._sts_det.update(
            p_sum, p_knee, self._sm.state, self._knee_baseline, t,
        )
        if post_sts:
            candidate = "STANDING_UPRIGHT"
        elif force_sts:
            candidate = "SIT_TO_STAND"
        else:
            candidate = cand

        immediate = force_sts or post_sts
        state = self._sm.propose(candidate, t, immediate=immediate)

        step_event = False
        if raw_step and state in self._STEP_STATES:
            step_event = True
            self._counters["total_steps"] += 1
            if state == "STAIRS_UP":
                self._counters["up_steps"] += 1
            elif state == "STAIRS_DOWN":
                self._counters["down_steps"] += 1
            elif state == "WALKING_FORWARD":
                self._counters["forward_steps"] += 1
            elif state == "WALKING_BACKWARD":
                self._counters["backward_steps"] += 1

        dbg = {
            "p_toe":  round(p_toe, 3),
            "p_ff":   round(p_ff, 3),
            "p_heel": round(p_heel, 3),
            "p_knee": round(p_knee, 3),
            "p_sum":  round(p_sum, 3),
            "left_load":          round(p_sum, 3),
            "right_load":         0.0,
            "lr_ratio":           0.0,
            "last_valid_walk_dir": self._last_valid_walk_dir,
            "step_source":        step_source,
            "left_foot_phase":    self._contact_det_single.phase,
            "right_foot_phase":   None,
            "raw_dir":  per_step_dir,
            "layer1_branch":      layer1,
            "layer2_subbranch":   layer2,
            "branch_rf_key":      br_key,
            "branch_rf_file":     rf_name,
            "ml_label":           cand,
            "knee_min_raw":       round(min_raw_knee, 1),
            "layer2_reason":    layer2_reason,
            "rf_proba":           round(rf_p, 4),
            "rf_reject":          rf_rej,
            "sts_phase": self._sts_det.phase,
            "sts_trigger_level": round(self._sts_det.trigger_level, 3),
            "sts_confirm_level": round(self._sts_det.confirm_level, 3),
            "ml_feature_mode": "branch_adaptive_v2",
        }
        dbg.update(knee_gate_info)
        dbg.update(sign)
        if adaptive_dbg:
            dbg["adaptive"] = adaptive_dbg
        return {
            "state":              state,
            "step_event":         step_event,
            "walk_dir":           walk_dir,
            "counters":           dict(self._counters),
            "sts_last_duration_s": self._sts_det.last_duration,
            "debug": dbg,
        }

    def update_bilateral(
        self,
        raw_left: tuple[float, float, float, float],
        raw_right: tuple[float, float, float, float],
        t: float | None = None,
    ) -> dict:
        """
        双足融合：左脚 (toe, forefoot, heel, knee)，右脚同样顺序。
        迈步主要靠脚底接触检测（抬起 → 落地）按脚走。
        """
        if t is None:
            t = time.monotonic()

        lt, lf, lh, lk = raw_left
        rt, rf, rh, rk = raw_right

        adaptive_dbg: dict[str, dict] = {}
        raw8 = np.array([lt, lf, lh, lk, rt, rf, rh, rk], dtype=np.float64)
        raw8_calib = self._calibrate_raw8(raw8)
        _flat24, snaps = self._adapt_bank.update(raw8_calib)
        self._last_adaptive_snaps = list(snaps)
        p_toe_l = self._f_toe.update(snaps[0].relative_pressure_ratio)
        p_ff_l = self._f_ff.update(snaps[1].relative_pressure_ratio)
        p_heel_l = self._f_heel.update(snaps[2].relative_pressure_ratio)
        p_knee_l = self._f_knee.update(snaps[3].relative_pressure_ratio)
        p_toe_r = self._f_toe_r.update(snaps[4].relative_pressure_ratio)
        p_ff_r = self._f_ff_r.update(snaps[5].relative_pressure_ratio)
        p_heel_r = self._f_heel_r.update(snaps[6].relative_pressure_ratio)
        p_knee_r = self._f_knee_r.update(snaps[7].relative_pressure_ratio)
        self._p_hist.append(_flat24)
        for nm, sn in zip(CHANNEL_NAMES_DUAL, snaps):
            adaptive_dbg[nm] = _snapshot_to_adaptive_debug_dict(sn)

        left_load = p_toe_l + p_ff_l + p_heel_l
        right_load = p_toe_r + p_ff_r + p_heel_r
        p_sum = left_load + right_load
        p_knee_avg = 0.5 * (p_knee_l + p_knee_r)
        lr_ratio = (left_load - right_load) / (left_load + right_load + 1e-9)

        if self._knee_baseline is None:
            self._knee_init_buf.append(p_knee_avg)
            if len(self._knee_init_buf) >= 30:
                self._knee_baseline = float(np.mean(self._knee_init_buf))
        min_raw_knee = self._compute_min_knee_raw(lk, rk)
        layer1, knee_gate_info = self._update_layer1_knee_gate(min_raw_knee, t)
        # 膝盖伸直见证：两只膝盖里只要有一只摆到 4095 轨就算（摆腿阶段）
        knee_extended_now = max(float(lk), float(rk)) >= WALK_KNEE_EXTEND_RAIL_TH
        dom_left = left_load >= right_load
        dom_toe = p_toe_l if dom_left else p_toe_r
        dom_ff = p_ff_l if dom_left else p_ff_r
        dom_heel = p_heel_l if dom_left else p_heel_r
        self._update_zone_sequence(dom_toe, dom_ff, dom_heel, t)

        # 主用：脚底接触迈步检测
        step_l = self._contact_det_l.update(left_load, t)
        step_r = self._contact_det_r.update(right_load, t)
        contact_step = step_l or step_r
        step_source: str | None = None
        if step_l:
            step_source = "left_contact"
        elif step_r:
            step_source = "right_contact"

        # 备用：脚跟版迈步检测（fallback + 调试）
        sl = self._heel_step_l.update(p_heel_l)
        sr = self._heel_step_r.update(p_heel_r)
        heel_step_l = self._step_det_l.update(sl)
        heel_step_r = self._step_det_r.update(sr)
        heel_step = heel_step_l or heel_step_r

        raw_step = contact_step
        if not raw_step and heel_step:
            raw_step = True
            step_source = "heel_fallback"

        if raw_step:
            self._last_step_t = t
            self._dir_det.on_step(t)

        # 方向判断
        per_step_dir = self._dir_det.feed(
            t, max(p_toe_l, p_toe_r), max(p_heel_l, p_heel_r),
        )
        if per_step_dir is not None:
            self._dir_vote.push(per_step_dir)
        walk_dir = self._dir_vote.result
        if walk_dir in ("forward", "backward"):
            self._last_valid_walk_dir = walk_dir

        recent_step = (t - self._last_step_t) < WALK_TIMEOUT_S
        amp = self._estimate_motion_amplitude(
            max(p_toe_l, p_toe_r),
            max(p_ff_l, p_ff_r),
            max(p_heel_l, p_heel_r),
            p_knee_avg,
        )
        layer2, layer2_reason = self._layer2.update(
            raw_step,
            recent_step,
            amp,
            t,
            layer1_branch=layer1,
            knee_extended_now=knee_extended_now,
        )
        sign = self._gait_signature_from_pressures(
            dom_toe, dom_ff, dom_heel, p_knee_avg,
        )
        br_key = self._branch_key_from_layers(
            layer1 == "ACTIVE_BRANCH", layer2,
        )
        cand, rf_p, rf_rej, rf_name = self._hierarchical_rf_predict(
            br_key,
            sign,
            left_load,
            right_load,
            lr_ratio,
            p_sum,
        )

        force_sts, post_sts = self._sts_det.update(
            p_sum, p_knee_avg, self._sm.state, self._knee_baseline, t,
        )
        if post_sts:
            candidate = self._classify_standing_bilateral(left_load, right_load)
        elif force_sts:
            candidate = "SIT_TO_STAND"
        else:
            candidate = cand

        immediate = force_sts or post_sts
        state = self._sm.propose(candidate, t, immediate=immediate)

        # 计步只在行走 / 爬楼这几个状态下累加，站着或坐着的抖动不算
        step_event = False
        if raw_step and state in self._STEP_STATES:
            step_event = True
            self._counters["total_steps"] += 1
            if state == "STAIRS_UP":
                self._counters["up_steps"] += 1
            elif state == "STAIRS_DOWN":
                self._counters["down_steps"] += 1
            elif state == "WALKING_FORWARD":
                self._counters["forward_steps"] += 1
            elif state == "WALKING_BACKWARD":
                self._counters["backward_steps"] += 1

        th_l = self._step_det_l.thresholds
        th_r = self._step_det_r.thresholds
        dbg = {
            "p_toe_l":          round(p_toe_l, 3),
            "p_ff_l":           round(p_ff_l, 3),
            "p_heel_l":         round(p_heel_l, 3),
            "p_knee_l":         round(p_knee_l, 3),
            "p_toe_r":          round(p_toe_r, 3),
            "p_ff_r":           round(p_ff_r, 3),
            "p_heel_r":         round(p_heel_r, 3),
            "p_knee_r":         round(p_knee_r, 3),
            "p_sum":            round(p_sum, 3),
            "left_load":        round(left_load, 3),
            "right_load":       round(right_load, 3),
            "lr_ratio":         round(lr_ratio, 3),
            "last_valid_walk_dir": self._last_valid_walk_dir,
            "step_source":      step_source,
            "left_foot_phase":  self._contact_det_l.phase,
            "right_foot_phase": self._contact_det_r.phase,
            "heel_combined":    round(max(sl, sr), 3),
            "heel_smooth_l":    round(sl, 3),
            "heel_smooth_r":    round(sr, 3),
            "step_th_l":        (round(th_l[0], 3), round(th_l[1], 3)),
            "step_th_r":        (round(th_r[0], 3), round(th_r[1], 3)),
            "raw_dir":          per_step_dir,
            "layer1_branch":    layer1,
            "layer2_subbranch": layer2,
            "branch_rf_key":    br_key,
            "branch_rf_file":   rf_name,
            "ml_label":         cand,
            "knee_min_raw":     round(min_raw_knee, 1),
            "layer2_reason":    layer2_reason,
            "rf_proba":         round(rf_p, 4),
            "rf_reject":        rf_rej,
            "sts_phase":        self._sts_det.phase,
            "sts_trigger_level": round(self._sts_det.trigger_level, 3),
            "sts_confirm_level": round(self._sts_det.confirm_level, 3),
            "ml_feature_mode":  "branch_adaptive_v2",
        }
        dbg.update(knee_gate_info)
        dbg.update(sign)
        if adaptive_dbg:
            dbg["adaptive"] = adaptive_dbg
        return {
            "state":              state,
            "step_event":         step_event,
            "walk_dir":           walk_dir,
            "counters":           dict(self._counters),
            "sts_last_duration_s": self._sts_det.last_duration,
            "debug": dbg,
        }

    # ── 内部辅助 ─────────────────────────────────────────────────────

    def _resolve_walk_direction(self) -> str | None:
        """返回当前有效的走向（'forward' / 'backward'），拿不到就 None。"""
        d = self._dir_vote.result
        if d in ("forward", "backward"):
            return d
        return self._last_valid_walk_dir

    def _is_motion_state(self, s: str) -> bool:
        return s in self._STEP_STATES

    def _estimate_motion_amplitude(
        self,
        toe: float,
        ff: float,
        heel: float,
        knee: float,
    ) -> float:
        vals = [toe, ff, heel, knee]
        inst = float(max(vals) - min(vals))
        self._motion_amp_hist.append(inst)
        if not self._motion_amp_hist:
            return inst
        return float(max(self._motion_amp_hist) - min(self._motion_amp_hist))

    def _gait_signature_from_pressures(
        self,
        toe: float,
        ff: float,
        heel: float,
        knee: float,
    ) -> dict[str, object]:
        total = max(toe + ff + heel, 1e-9)
        heel_impact = heel / total
        forefoot_dom = (toe + ff) / total
        knee_activity = (
            abs(knee - self._knee_baseline)
            if self._knee_baseline is not None
            else knee
        )

        initial, contact_order, release_order, complete = self._extract_gait_orders()

        return {
            "initial_contact_zone": initial,
            "contact_order": contact_order,
            "release_order": release_order,
            "heel_impact_score": float(heel_impact),
            "forefoot_dominance_score": float(forefoot_dom),
            "knee_activity_level": float(knee_activity),
            "gait_signature_complete": bool(complete),
        }

    def _update_zone_sequence(self, toe: float, ff: float, heel: float, t: float) -> None:
        vals = {"toe": toe, "forefoot": ff, "heel": heel}
        for zone, v in vals.items():
            prev = self._zone_contact_state[zone]
            now = prev
            if prev:
                if v <= FOOT_OFF_GROUND_TH:
                    now = False
            else:
                if v >= FOOT_ON_GROUND_TH:
                    now = True
            if now != prev:
                self._zone_contact_state[zone] = now
                self._zone_event_hist.append(("on" if now else "off", zone, float(t)))

    def _extract_order(self, kind: str) -> list[str]:
        seq: list[str] = []
        seen: set[str] = set()
        for evt, zone, _t in reversed(self._zone_event_hist):
            if evt != kind:
                continue
            if zone in seen:
                continue
            seq.append(zone)
            seen.add(zone)
            if len(seq) >= 3:
                break
        seq.reverse()
        return seq

    def _extract_gait_orders(self) -> tuple[str, str, str, bool]:
        contact = self._extract_order("on")
        release = self._extract_order("off")
        if len(contact) < 3 or len(release) < 3:
            return "unknown", "unknown", "unknown", False
        contact_order = "->".join(contact)
        release_order = "->".join(release)
        initial = contact[0]
        complete = len(set(contact)) == 3 and len(set(release)) == 3
        return initial, contact_order, release_order, complete

    def _classify_standing_bilateral(
        self, left_load: float, right_load: float,
    ) -> str:
        """根据左右脚承重分布判断：直立 / 偏左 / 偏右。"""
        lr_ratio = (left_load - right_load) / (left_load + right_load + 1e-9)
        if abs(lr_ratio) < LEAN_LR_TH:
            return "STANDING_UPRIGHT"
        if lr_ratio > LEAN_LR_TH:
            return "STANDING_LEFT_LEAN"
        return "STANDING_RIGHT_LEAN"
