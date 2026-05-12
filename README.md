# 智能袜子 · 姿态识别系统（分级级联 · 四独立 RandomForest）

> 双脚 × 4 通道压阻/拉伸传感器（`toe / forefoot / heel / knee`），**10 Hz** 采样（固件每 100 ms 发一帧，已按 CSV `Timestamp` 列复核），
> **Layer‑1 膝盖 4095 规则** → **Layer‑2 motion/static** → **4 个完全独立的 RF**
> → 状态机防抖输出 9 类姿态 + `SIT_TO_STAND` + `UNKNOWN`。

本仓库**只保留四类核心内容**，老旧实验产物已全部清理。所有讲解、训练、推理试跑都走
`smart_sock_pipeline.ipynb` **一个** notebook。

---

## 目录结构（清理后）

```
monitor/                 # 工程根（只剩智能袜子系统）
├── smart_sock_pipeline.ipynb               # 教学型主 notebook（采集 → 预处理 → 个人标定 → 特征 → 训练 → 推理）
├── foot_pressure_monitor.py                # UI 上位机（纯 PyQt5 瘦身版：socket 实时数字 + 两级级联 HUD + 采集标签 + Calibration 向导）
├── personal_calibration.py                 # 【新】个人量程标定：离线自动 + 在线 UI 两条路径，同一数据类
├── adaptive_preprocessing.py               # 在线 EWMA 自适应预处理（8 通道独立）
├── ml_activity_features.py                 # 特征工程 + CSV 加载 + 滑窗构建（可选择是否注入标定）
├── ml_branch_models.py                     # 四分支 RF bundle 加载与 aux 向量
├── ml_train_branch_rfs.py                  # 训练脚本：按标签分桶训练 4 只独立 RF，默认先做离线标定
├── realtime_recognizer.py                  # 在线识别：Layer1 + Layer2 + 4 RF + 状态机（支持 `calibration=…`）
├── personal_calibration.json               # 运行训练脚本或 UI 标定后产生 → 被推理端自动热加载
├── requirements.txt                        # 最小依赖
├── saving_data/
│   ├── sensor_data_dual_labeled_*.csv      # 用于训练的带标签 CSV（英文规范命名）
│   └── archive_old/                        # 旧版/中文命名 CSV 归档（不参与训练）
└── README.md                               # 本文件
```

训练完生成 4 个 `.joblib`（**当前是被删光的，跑训练后重新生成**）：

- `rf_active_motion.joblib`    → `STAIRS_UP / STAIRS_DOWN`
- `rf_active_static.joblib`    → `SITTING_NORMAL / SITTING_CROSSLEGGED`
- `rf_inactive_motion.joblib`  → `WALKING_FORWARD / WALKING_BACKWARD`
- `rf_inactive_static.joblib`  → `STANDING_UPRIGHT / STANDING_LEFT_LEAN / STANDING_RIGHT_LEAN`

---

## 运行顺序（推荐）

```
 [环境安装] ─► [启动 UI 采集数据] ─► [notebook 检查预处理 + 特征] ─► [notebook / CLI 训练 4 RF]
                                                                     │（默认产出 personal_calibration.json）
                                                                     ▼
           [UI 切到 Calibration 为当前佩戴者做两步 1.站立 2.弯膝 90° 标定 → 热重载 Recognizer]
                                                                     │
                                                                     ▼
                                                          [UI 切到 Inference 实时试跑]
```

**数据集训练 vs 真人部署的切换**：
- 只想跑数据集 → 什么都不用做，训练脚本会自动扫 `saving_data/` 生成 `personal_calibration.json`。
- 真人部署 → 打开 UI，走一次 **Calibration** 向导，它会覆盖同一个 JSON 并热重载识别器。
- 两种模式**共用同一份 JSON 文件格式**，无需改任何代码即可互换。

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install scikit-learn joblib matplotlib
```

`requirements.txt` 里的最小依赖只保证 UI + 训练能跑：`PyQt5`, `numpy`, `scikit-learn`, `joblib`。
notebook 里画图时再额外要 `matplotlib`。UI 已经不再依赖 `pyqtgraph`。

### 2. 数据采集

```bash
python foot_pressure_monitor.py
```

- 在 UI 里把 **左脚端口** 设成 `5000`、**右脚端口** 设成 `6000`（默认即是）。
- 切到 **Data capture** 标签 → **Start** → 选动作按钮打标签（同一 CSV 段只做一种动作）→ **Stop**。
- 每次 Start/Stop 会在 `saving_data/` 生成两份 CSV：
  - `sensor_data_dual_labeled_<时间戳>.csv`（含 `Label` 列，**训练用**）
  - `sensor_data_dual_raw_<时间戳>.csv`（纯备份，无标签）
- 每类动作建议**连续 2 分钟**以上，多人 / 多场次 / 换鞋垫多采几次。

### 3. 个人量程标定（Personal Calibration）+ 全局统计归一化

个人标定是介于「raw ADC」和「RF 特征」之间的一层**硬归一化**，由两块组成：

1. **`[min_raw, max_raw]` 线性重映射**：为每个通道学一条静态工作区间，把 raw 线性地重映射到统一的 `[0, 4095]`。
2. **全局统计冻结 EWMA**（v2 新增）：再算出 `baseline_raw / press_min / press_max / press_mean / press_std` 并**冻结** `adaptive_preprocessing` 的 EWMA，使每个 CSV 的每一帧都用同一套参数做 `baseline_removed / relative_pressure_ratio / adaptive_zscore`，彻底消除逐文件 / 前 N 帧局部统计带来的漂移。

**离线 / 在线两种来源产生同一份 JSON**，字段也完全一致。

#### 3.a 离线自动标定（训练阶段）

默认训练脚本就会干这件事：

```bash
python ml_train_branch_rfs.py                    # 自动扫 saving_data 产生 personal_calibration.json 并用它训练
python ml_train_branch_rfs.py --calib my_ui.json # 复用一份 UI 向导生成的个人标定
python ml_train_branch_rfs.py --no-calib         # 消融：完全不做个人标定
```

离线标定规则：

- **阶段 A · min/max**（线性重映射区间）：
  - 压力垫（toe / forefoot / heel × L/R）：
    - `min_raw` = 在 `WALKING_*` / `STAIRS_*` / `STANDING_*` 段取 **1 分位点** → 个人满载脚趾/脚跟 peak。
    - `max_raw` = 在 `SITTING_*` 段取 **97 分位点** → 无负载参考。
  - 膝盖拉伸（L_Knee / R_Knee）：
    - `min_raw` = 在 `SITTING_*` / `STAIRS_*` 段取 **1 分位点** → 最深弯曲。
    - `max_raw` = `4095`（伸直参考）。
  - 若某通道动态范围小于 `OFFLINE_PRESSURE_MIN_SPAN (120)` 或 `OFFLINE_KNEE_MIN_SPAN (300)`，自动回退 `[0, 4095]`，避免把特征全压扁。
- **阶段 B · 全局统计（v2 新增，跨 33 份 CSV）**：
  - `load_csv_files` 先把 `saving_data/sensor_data_dual_labeled_*.csv` 33 份 labeled CSV **全部拼**到一个 `(T, 8)` 整体矩阵（`T ≈ 2 万帧`）——禁止逐文件、禁止前 N 行截断。
  - `baseline_raw[ch] = max_raw[ch]`（空载参考）；`press_mag = baseline_raw − raw`（≥ 0）。
  - `press_min[ch] / press_max[ch]` = 整体矩阵上 `press_mag` 的 1% / 99% 分位点；若 `press_max − press_min < 1`，回退 `[0, 4095]`。
  - `press_mean[ch] / press_std[ch]` = 整体矩阵上 `press_mag` 的均值 / 标准差（std 下限 1.0，防除零）。
  - 这 5 个字段通过 `PersonalCalibration.to_channel_seeds()` 打包成 8 个 `adaptive_preprocessing.ChannelSeed`，传给 `DualFootAdaptiveBank(seeds=...)`，让每个 `AdaptiveSensorPreprocessor` **全程冻结 EWMA**。

#### 3.b UI 在线两步标定（正式部署）

```bash
python foot_pressure_monitor.py      # 启动 UI
```

1. 上排连通 MCU 后，切到 **Calibration** 标签。
2. 填写 `Subject` 名字。
3. **Start Step 1** → 穿戴者**自然直立站立 5 秒** → **Finish Step 1**（采压力范围）。
4. **Start Step 2** → 穿戴者**双膝弯 90° 保持 5 秒** → **Finish Step 2**（采拉伸范围）。
5. 预览框检查 8 通道 min/max/span 合理 → **Save personal_calibration.json + reload**。
6. 主窗口会**立即热重载** `OnlineRecognizer`，下一帧就是新个人量程，无需重启。

#### 3.c `personal_calibration.json` 文件格式

```json
{
  "channels": ["L_Toe","L_Forefoot","L_Heel","L_Knee","R_Toe","R_Forefoot","R_Heel","R_Knee"],
  "min_raw":      [0.0, 1792.0, 1088.4, 31.0, 2390.4, 2132.0, 2145.4, 179.0],
  "max_raw":      [4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095],
  "baseline_raw": [4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095],
  "press_min":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "press_max":    [4095, 2258, 3012, 4047, 1849, 1976, 1942, 3894],
  "press_mean":   [2128, 824, 1683, 961, 893, 935, 663, 442],
  "press_std":    [1499, 716, 1161, 1565, 634, 676, 664, 940],
  "source":   "offline_auto_csv",   // 或 "ui_online_two_step"
  "subject":  "default",
  "created_at": 1718000000.0,
  "notes": "..."
}
```

若 JSON 缺少后 5 个字段（老版本），`DualFootAdaptiveBank` 会自动退回到 EWMA 行为，**前后兼容**。

#### 3.d 约束（必看）

- **Layer-1 膝盖门控始终用原始 raw**。`KNEE_RAW_STRAIGHT_TH = 3500` 作用于真实 ADC，**不受任何标定影响**。单腿非 4095 → ACTIVE、双腿 4095 → INACTIVE 的铁律不变。
- **必须全集统计**。`OfflineAutoCalibrator.fit` 接收的 `(T, 8)` 必须是全部 labeled CSV 拼起来的整体矩阵；禁止逐文件局部归一化、禁止用前 N 行数据估计统计量——否则 RF 会把局部分布当成特征学进去。
- **变了 RF 特征尺度就必须重训**。换一份新的 `personal_calibration.json` 想让 4 只 RF 真正受益，就 `python ml_train_branch_rfs.py --calib personal_calibration.json` 重训。UI 热重载只是把识别器上的标定换掉，**模型本身还是原来那组**。
- **调试时可先跑 `python -c "import personal_calibration as pc; print(pc.auto_calibrate_from_csv_dir('saving_data').summary())"`** 快速看一眼当前数据集学出来的个人量程和全局统计长什么样。

### 4. 跑 notebook 做预处理检查 + 训练 4 个独立 RF

用 Jupyter / VSCode 打开 **`smart_sock_pipeline.ipynb`**，按顺序逐格运行：

- **第二章 · Preprocessing**：看看归一化和自适应到底做了什么。
- **第三章 · 特征**：`build_dataset_adaptive` 生成 `(N, 347)` 的训练矩阵。
- **第四–五章 · 级联门控**：在一条真实 CSV 上复盘 Layer‑1 / Layer‑2 决策。
- **第六章 · 训练**：一个 cell 就会跑完 `ml_train_branch_rfs.main()`，根目录出现 4 个 `rf_*.joblib`。
- **第七章 · 推理**：把同一条 CSV 喂给 `OnlineRecognizer`，对照 Layer1/Layer2/final state 的完整轨迹。

> **也可以命令行一条命令训练**：`python ml_train_branch_rfs.py`

### 5. 实时推理

UI 切到 **Inference** 标签 → **Start**。上方 HUD 显示：

- `STATE: …`              最终识别结果
- `KNEE L1: …`            Layer‑1 分支 + phase + `min_raw`（当前两脚膝盖 raw 的较小值）
- `DIR / STEPS / …`       方向、步数、上下楼步数、坐→站耗时
- `DEBUG`                  walking guard 计数、RF proba / reject、gait signature

> Inference 模式**不会写 CSV**。

---

## 预处理方法（核心原理）

把原始 `raw ADC (0–4095)` 直接塞给 RF 会**严重过拟合硬件偏差**（每通道电阻不一样 → 基线不一样 → 量程不一样），所以**三件事**缺一不可（两层预处理 + 一层 per-subject 个人标定）：

| 步骤 | 代码 | 作用 | 参数更新 |
|------|------|------|----------|
| **Normalization 归一化** | `(SENSOR_MAX - raw) / SENSOR_MAX` 以及管线里的 `StandardScaler` | 把原始 ADC 压到 `[0,1]`，再把特征向量统一到零均值单位方差 | **常数 / 训练集估计** |
| **Personal Calibration 个人标定**【新】 | `personal_calibration.PersonalCalibration.normalize_to_adc` | 把每个通道 **`[personal_min, personal_max]` 重映射到 `[0, 4095]`**，消除体重 / 脚型 / 袜子绑带差异 | **离线从 CSV 学 / 在线 UI 两步标定**（见 §3） |
| **Adaptive Calibration 自适应自校准** | `adaptive_preprocessing.AdaptiveSensorPreprocessor`（**每通道一个实例**） | 在线估计每通道的 **baseline** 和 **dynamic_min/max**，输出硬件无关的 `relative_pressure_ratio`、`adaptive_zscore` | **在线 EWMA** |

> 个人标定是**静态（每人一次）**，自适应自校准是**动态（每帧都在更新）**。前者消除「你和别人不一样」，后者消除「你现在这几分钟和 5 分钟前不一样」。

### 自适应三件输出

对每个物理通道、每一帧输出：

- `baseline_removed = raw − baseline_raw`（带符号 delta）
- `relative_pressure_ratio ∈ [0,1]`（**主特征**，硬件无关）
- `adaptive_zscore`（辅助，短期归一化）

基线更新是**非对称**的：

- **空载** → `BASELINE_ALPHA_IDLE = 0.015` 快速追
- **轻压** → `BASELINE_ALPHA_SLOW = 0.002` 慢追
- **长时间持续压力** → `BASELINE_ALPHA_HEAVY_FREEZE = 0.0002`（**冻结**下拉，防止长时间坐/站把基线吃低）
- **卸载 (raw > baseline)** → `BASELINE_ALPHA_RECOVER = 0.03` 快速回升

### 为什么必须三层都做

- **只归一化**：`(4095 - raw) / 4095` 没抵消通道基线偏移，换人 / 换袜子 RF 直接崩。
- **只自适应**：`baseline_removed` 带量纲，`adaptive_zscore` 不保证有界，RF 端仍需要 `StandardScaler` 归一化；而且 EWMA 只能抓**短期**漂移，同一个人不同体重 / 不同鞋垫的**静态**差异需要个人标定补。
- **只个人标定**：静态量程对齐没法处理袜子滑动、绑带随时间松弛等**动态**问题，仍然需要 EWMA 在线追踪。

顺序固定：**raw → 个人标定 → 自适应 EWMA → 窗口统计/频域特征 → StandardScaler → RandomForest**。（Layer-1 膝盖 4095 门控始终并行读原始 raw。）

---

## 训练流程

由 `ml_train_branch_rfs.py` / notebook 第六章调度，步骤：

1. `load_csv_files(DATA_DIR, labeled_only=True, raw_adc=True)` 读入所有 `sensor_data_dual_labeled_*.csv`。
2. **个人标定**：`OfflineAutoCalibrator().fit(raw, labels)` 扫整份数据学出 8 通道 `(min_raw, max_raw)`，保存到 `personal_calibration.json`。可以用 `--calib path.json` 换成 UI 向导生成的个人版本，或者用 `--no-calib` 关掉（消融用）。
3. `simulate_adaptive_sequence_dual(raw, calibration=calib)` —— 标定后的 raw 被**因果地**回放一遍，得到 `(T, 8, 3)` 自适应特征序列（与在线推理**逐帧一致**）。
4. `WINDOW_SIZE = 10`（1 s @ 10 Hz）、`WINDOW_STEP = 2`（0.2 s，80 % 重叠）做滑动窗口，**80 % 多数标签一致**才作为样本。窗口压到 1 s 是为了在线推理延迟更低；若之后固件把采样率从 10 Hz 改成其它值，请同步改 `ml_activity_features.SAMPLE_HZ` 和 `realtime_recognizer.SAMPLE_HZ`（二者必须一致），`WINDOW_SIZE/STEP` 会按 `ML_WINDOW_DURATION_S = 1.0 s` 自动折算，改完必须重训 4 只 RF。
5. `LABEL_TO_BRANCH` 把每个样本按标签映射到 4 个桶之一：
   ```python
   STAIRS_UP / STAIRS_DOWN            → active_motion
   SITTING_NORMAL / SITTING_CROSSLEGGED → active_static
   WALKING_FORWARD / WALKING_BACKWARD → inactive_motion
   STANDING_UPRIGHT / _LEFT_LEAN / _RIGHT_LEAN → inactive_static
   ```
6. 每桶**独立** `train_test_split(test=0.25, stratify)`；每桶独立 `Pipeline(StandardScaler + RandomForestClassifier(n_estimators=200, max_depth=22, class_weight=balanced_subsample))`；各自 `joblib.dump` 成独立 bundle，bundle 里还写入 `calibration` 段记录训练时用了哪份 `[min_raw, max_raw]`。
7. 输出 4 个 `rf_*.joblib`，包含 `pipeline / branch / classes / aux_dim / feature_mode / calibration`。

**4 个模型完全独立**：它们不共用训练集、不共用特征分支、不共用预测。运行时由上游门控指定**任意时刻只有一只 RF 在推理**。

---

## 推理使用步骤（逻辑细节）

每帧 `OnlineRecognizer.update_bilateral(raw_L_4, raw_R_4, t)` 流程：

1. **自适应预处理**：8 通道送进 `DualFootAdaptiveBank`，拿到 `flat24`（8 通道 × `[br, ratio, zscore]`）和 `snaps`。
2. **滑动窗口**：把 `flat24` 追加进 `_p_hist`（长度 `≥ WINDOW_SIZE`）。
3. **Layer‑1 膝盖门控（严格 4095 规则）**：
   - `min_raw_knee = min(raw_knee_l, raw_knee_r)`
   - 单帧 ACTIVE ⇔ `min_raw_knee < KNEE_RAW_STRAIGHT_TH = 3500`
   - **防抖**：`KNEE_GATE_SUSTAIN_SAMPLES = 16`、`KNEE_GATE_RELEASE_SAMPLES = 14`、`KNEE_GATE_MIN_HOLD_S = 0.6 s`
4. **真实抬脚检测**：`_FootContactStepDetector`（基于脚总载荷的 lift → land 事件），左右各一个。
5. **Layer‑2 motion / static**：`MOTION_CONFIRM_FRAMES = 4` + `MOTION_MIN_AMPLITUDE = 0.10` + `MOTION_MIN_HOLD_S = 1.0`。
   - **INACTIVE 下的走路强约束**（保留用户规范）：还需要 `WALK_ENTER_MIN_STEPS = 2` 个真实抬脚事件 **且** 近 `WALK_EVIDENCE_WINDOW_S = 2.0 s` 内至少 2 次 `max(raw_knee) ≥ 4080` 的摆动相见证，才允许 STATIC → MOTION。
6. **选 RF**：由 (Layer1, Layer2) 组合决定 `branch_rf_key ∈ {active_motion, active_static, inactive_motion, inactive_static}`。
7. **RF 推理**：对应 `BranchRFBundle.predict(feat_367)`；`max_proba < BRANCH_RF_PROBA_MIN (=0.45)` → `UNKNOWN`。
8. **坐→站规则**：`_SitToStandDetector` 仅靠 `p_sum / p_knee / 趋势` 给 `SIT_TO_STAND` 和 `post_complete_standing` 两个事件。
9. **状态机**：`_StateMachine.propose(candidate, t)`，`STATE_MIN_DURATION_S = 1.0 s` 的最小保持；坐→站给 `immediate=True`。

最终返回 `dict`：`state / step_event / walk_dir / counters / sts_last_duration_s / debug`。`debug` 里有 `layer1_branch / layer2_subbranch / branch_rf_key / rf_proba / rf_reject / knee_min_raw / knee_gate_phase / walk_guard_reason / adaptive[*]` 等排障信息。

---

## 阈值调节速查

> 全部参数集中在 `realtime_recognizer.py` / `adaptive_preprocessing.py` / `ml_branch_models.py` 顶部，都标了 `# TODO_PARAM`。

| 症状 | 参数 | 默认 | 方向 |
|------|------|------|------|
| 走路被错判成 Active / 上下楼 | `KNEE_RAW_STRAIGHT_TH` | 3500 | **减小**（如 3000） |
| 轻微弯腿没被判成 Active | `KNEE_RAW_STRAIGHT_TH` | 3500 | **增大**（如 3800） |
| Layer‑1 在站/坐边界闪 | `KNEE_GATE_SUSTAIN_SAMPLES`, `KNEE_GATE_RELEASE_SAMPLES`, `KNEE_GATE_MIN_HOLD_S` | 16 / 14 / 0.6 s | **增大** |
| 站着抖动被误判为 WALKING | `WALK_ENTER_MIN_STEPS`, `WALK_EVIDENCE_WINDOW_S` | 2 / 2.0 s | 步数**增大**；窗口**缩短** |
| 真走路迟迟不进 WALKING | `WALK_ENTER_MIN_STEPS` | 2 | **减小**（1） |
| 脚抬得不高没被识别 | `WALK_KNEE_EXTEND_RAIL_TH` | 4080 | **减小**（3950） |
| forward/backward 反了 | `WINDOW_SECONDS`, `VOTE_K_STEPS` | 0.6 s / 3 | 按数据调 |
| 跨腿坐识别不灵 | `SIT_CROSSLEG_STD_TH`, `SIT_CROSSLEG_KNEE_TH` | 0.04 / 0.15 | 按数据调 |
| 坐→站漏/误触发 | `STS_TRIGGER_TH`, `STS_CONFIRM_TH`, `STS_CONFIRM_S` | 0.15 / 0.25 / 0.8 | 按数据调 |
| RF 经常 UNKNOWN | `BRANCH_RF_PROBA_MIN`（`ml_branch_models.py`） | 0.45 | **减小**（0.35） |
| 自适应基线飘太慢/太快 | `BASELINE_ALPHA_SLOW/IDLE/RECOVER/HEAVY_FREEZE`（`adaptive_preprocessing.py`） | — | 按通道特性调 |

**改了 `SAMPLE_HZ / ML_WINDOW_DURATION_S / WINDOW_SIZE / WINDOW_STEP / aux_dim / 特征构成` 必须重新训练全部 4 个 RF**，否则特征维度会对不上。

---

## 与之前版本的差异（已清理）

本次重构的动作清单：

- 高考志愿填报网站（`backend/`, `frontend/`, `local-version/`, 根 `package.json`, 老 `README`）已整体移到 **`../gaokao-web-backup/`**，与本工程分离。
- 所有 `__pycache__`、`.ipynb_checkpoints`、`.idea` 缓存/IDE 配置清空。
- 老 `joblib`（`rf_*.joblib` 4 个）删除，notebook 跑一次训练即可重新生成。
- 过时 ipynb（`final.ipynb`, `ml_train_and_export_upgrade2.ipynb`, `guide_*.ipynb`, `update_explanation.ipynb`）删除，统一入口 `smart_sock_pipeline.ipynb`。
- 非核心脚本（`make_unlabeled_copies.py`, `offline_replay_eval.py`）、评测产物 `replay_eval_test/` 删除。
- `saving_data/` 下中文命名的老 CSV 归档到 `saving_data/archive_old/`，不再参与训练。
- `realtime_recognizer.py` 重构：
  - 把原来基于 `max(|raw − baseline|) ≥ 3900` 的膝盖门控改成**严格 4095 规则**：`min(raw_knee_l, raw_knee_r) < KNEE_RAW_STRAIGHT_TH`。
  - 防抖加入 **min‑hold timer** `KNEE_GATE_MIN_HOLD_S`，避免站立时 Layer1 闪烁。
  - Layer‑2 走路切换加入 **真实抬脚事件 + 摆动相 4095 见证 + 连续 2 步**的强约束（仅在 INACTIVE 下启用）。
  - UI 调试行改显 `min_raw` + `knee_gate_phase(+locked_*)`。
- **采样率从原先代码里默认的 40 Hz 校正为实测 10 Hz**（用 CSV `Timestamp` 列逐行核对两帧相差 100 ms）；随后**滑窗进一步从 3 s / 7 步压到 1 s / 2 步**（`ML_WINDOW_DURATION_S = 1.0 s` → `WINDOW_SIZE = 10`、`WINDOW_STEP = 2`），在线推理延迟直接降到 0.2 s 级别，4 个 RF 已按新窗口重训（测试准确率仍在 98–100 %）。
- **UI 彻底瘦身**：`foot_pressure_monitor.py` 去掉 `pyqtgraph` 热力图、直方图对话框、定时重绘这些卡顿源，只保留 socket 实时数字读数、Layer-1 / Layer-2 / RF HUD、数据采集标签条 和 Calibration 向导四块，跑起来不再卡。
- **新增个人量程标定层** `personal_calibration.py`：
  - `OfflineAutoCalibrator`：扫 labeled CSV 自动学出 8 通道 `(min_raw, max_raw)` → `personal_calibration.json`。
  - `OnlineCalibrator`：两步 UI 向导（Step 1 站立、Step 2 弯膝 90°）→ 同一个 JSON 格式。
  - `simulate_adaptive_sequence_dual(raw, calibration=…)` 和 `OnlineRecognizer(calibration=…)` 都支持透传同一个 `PersonalCalibration` 对象，**离线训练和在线推理接口一致**。
  - Layer-1 膝盖门控**仍然读原始 raw**，严格 4095 铁律不受任何标定影响。
- `foot_pressure_monitor.py` 增加 **Calibration** 选项卡：
  - 两步向导 + 进度条 + 预览。
  - 保存 JSON 时**热重载** `OnlineRecognizer`，下一帧立即用新标定，无需重启程序。
- `ml_train_branch_rfs.py` 默认就做一次离线个人标定再训 4 只 RF；通过 `--calib` / `--no-calib` 切换数据来源或消融。

---

## 依赖说明

- Python ≥ 3.10
- `PyQt5 ≥ 5.15`（UI；已移除对 `pyqtgraph` 的依赖）
- `numpy ≥ 1.21`
- `scikit-learn`（训练/推理）
- `joblib`（模型序列化）
- `matplotlib`（notebook 可视化）

硬件假设：双脚各 1 块 MCU，每帧 4 个 12 bit ADC，**10 Hz（每 100 ms 发一帧）**，按行 TCP 发送 `toe, forefoot, heel, knee`。如果将来固件上调到更高帧率，只需同步改 `ml_activity_features.SAMPLE_HZ` 与 `realtime_recognizer.SAMPLE_HZ` 并重训 4 只 RF。
