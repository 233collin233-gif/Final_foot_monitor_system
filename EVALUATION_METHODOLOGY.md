# 评测方法说明 — validation / test 是怎么测出来的

> 本文档回答一个问题："你的 4 只随机森林给出的 `mean / std / OOF accuracy` 到底是怎么算出来的？"
>
> 回答：**用 Blocked intra-file CV（分段块状交叉验证）** 在全部 33 份 labeled CSV 上做 5 折，最终部署的 `rf_*.joblib` 再用 100% 数据 fit 一次。
>
> 相关代码：`ml_train_branch_rfs.py`（`_assign_blocked_folds` / `BlockedIntraFileCV` / `_train_one_branch`）。

---

## 0. TL;DR — 一张图看完

```
                       33 份 labeled CSV
                       (时间顺序原封不动)
                              │
                              ▼
         ┌─── 逐文件加载 (_load_csv_per_file) ───┐
         │  每份 CSV 独立做 AdaptiveBank 冻结    │
         │  preprocessing (全局统计 seed)        │
         └───────────────┬───────────────────────┘
                         │
                         ▼
       滑动窗口 (WINDOW_SIZE=10, WINDOW_STEP=2)
       每个窗口打上 3 个标签：
         · branch         —— 分到哪只 RF
         · file_idx       —— 来自哪个 CSV
         · segment_id     —— "同文件同标签"的第几段
                         │
                         ▼
    ┌──── 按 branch 分桶，对每只 RF 单独做 ────┐
    │                                            │
    │  ① Blocked CV 评测                         │
    │     把该 branch 的 segment 各切 5 连续块   │
    │     fold k 的 test = 每段的第 k 块        │
    │     + 两侧 5 个窗口的 purge gap           │
    │     → 5 个 fold 准确率                     │
    │     → OOF 聚合预测（每窗口恰好被预测 1 次）│
    │     → 混淆矩阵（覆盖全部类别）             │
    │                                            │
    │  ② 最终模型 fit on ALL windows             │
    │     → rf_<branch>.joblib                   │
    │                                            │
    └────────────────────────────────────────────┘
```

**所以我们报告的 validation / test 分数 = ① 里的 blocked CV 指标**，部署模型是 ② 里的全量 fit。

---

## 1. 背景：为什么不能随便 split

### 1.1 我们没有"外部测试集"

全部训练数据来自 `saving_data/sensor_data_dual_labeled_*.csv`，共 **33 份 CSV，约 9871 个窗口**。没有额外的独立测试集可以用，validation / test 必须**从这 33 份里切出来**。

### 1.2 数据是 10 Hz 连续时间序列

- 固件每 100 ms 发 1 帧 → 相邻两帧**物理上几乎一样**（连续动作下压力、膝盖拉伸都只微小漂移）。
- 窗口大小 10 帧、步长 2 帧 → 相邻的两个**窗口**有 80% 重叠。
- ⚠️ 直接做"随机 shuffle"或"每个窗口独立 train/test split"都会让**几乎同一个物理状态**同时出现在 train 和 test 里 → 模型只要记住"上一个窗口是啥"就能拿到假分数，这不是泛化能力。

---

## 2. 候选评测方案及其问题

| 方案 | 做法 | 问题 |
|------|------|------|
| **A. 随机 shuffle** `train_test_split(shuffle=True)` | 所有窗口洗牌后切 60/20/20 | ❌ 相邻 100 ms 的窗口被分到 train/test 互相"偷看"，分数严重虚高（实测 `inactive_static` 曾跑到 99.0%） |
| **B. 按时间截前后** 最后 20% 当 test | 单文件内部时序保留 | ❌ 每份 CSV 只录 1–2 个标签 → test 集整体缺类，某些类的 recall 根本算不出来 |
| **C. Leave-One-File-Out (LOFO)** 留一份整 CSV 做 test | 彻底跨文件 | ⚠️ 同样因为单文件标签稀疏，**留出的那份常常只有 1–2 类**，该 fold 里其它类 recall = 0%；实测 `STANDING_UPRIGHT` LOFO recall = 0%、`inactive_static` std = 0.48 —— 方差吃掉参考价值 |
| **D. Blocked intra-file CV**（当前方案） | 每段 label-segment 内部切 K 连续块 | ✅ 每 fold test 覆盖全类；每块 test 是连续帧；purge gap 防边界泄漏；OOF 聚合每窗口预测 1 次 |

---

## 3. 当前方案：Blocked intra-file CV

### 3.1 核心定义

**Label-segment**：同一个 CSV 里、**同一个 majority label** 的**最长连续**窗口序列。

- 例：文件 `sensor_data_dual_labeled_07.csv` 里前 600 帧是 `SITTING_NORMAL`，后 500 帧是 `SITTING_CROSSLEGGED`，这就是**两个 segment**。
- 切分时 segment 之间相互独立。

### 3.2 算法

代码：`ml_train_branch_rfs.py::_assign_blocked_folds`

```
for each segment:
    n = segment 的窗口数
    if n < BLOCKED_N_FOLDS (=5):
        round-robin 分配（极短段兜底）
    else:
        用 np.linspace 把 [0, n] 等分成 5 连续块
        block k 的所有窗口 fold_id = k
```

**fold k 的 test 集** = 所有 segment 的第 k 块的并集。

### 3.3 Purge gap — 防滑窗重叠泄漏

代码：`BlockedIntraFileCV::split`

窗口步长 = 2 帧、窗口长度 = 10 帧 → 相邻 5 个窗口之间物理帧重叠。如果 test 块紧贴 train 块，test 块第 1 个窗口的前 80% 帧和 train 最后 1 个窗口的后 80% 帧是**同一段信号**。

解决：**对每个 test 窗口周围 `BLOCKED_PURGE = 5` 个位置（前后各 5）从 train 里 mask 掉。**

```
fold k:
    test_mask[i]   =  fold_ids[i] == k
    widened_mask   =  test_mask 向左/右各 shift 1..purge 次后按位或
    train_mask[i]  = (fold_ids[i] != k)  AND  not widened_mask[i]
```

这样 train 和 test 之间有 5 个窗口（= 10 个 step = 一整个窗口长度）的"真空地带"，完全不共享任何物理帧。

### 3.4 评测指标是怎么算的

每只 branch RF 走一次 CV，得到三组数：

| 字段 | 怎么算 | 含义 |
|------|--------|------|
| `blocked_fold_accuracies` | `sklearn.model_selection.cross_val_score(pipeline, X, y, cv=BlockedIntraFileCV, scoring='accuracy')` | **5 个数字**，每折在自己的 test 块上的准确率 |
| `blocked_mean / std / min / max` | 上面 5 个数的统计量 | **mean** 是每折 test 的未加权平均，**std** 是折间稳定性 |
| `blocked_n_segments` | 该 branch 里 label-segment 总数 | 辅助理解："这 N 段被各切成 5 块" |
| `oof_accuracy` | 先 `cross_val_predict` 拿到每个窗口的"被留出时的预测"，再 `accuracy_score(y, oof_pred)` | **按窗口加权的总准确率**，和 `mean` 通常差不多；OOF 聚合把"每窗口恰好被预测一次"的全部结果拼起来 |
| `confusion_oof` | `confusion_matrix(y, oof_pred, labels=classes_ordered)` | **覆盖全部类别的混淆矩阵**，行 = 真实类，列 = 预测类 |

### 3.5 最终部署模型

代码：`_train_one_branch` 最后几行

```python
pipe.fit(X, y)          # <── 全部窗口 fit
joblib.dump(bundle, rf_<branch>.joblib)
```

**CV 评测结束后，部署模型用 100% 窗口再 fit 一次**。原因：

- CV 每一折的模型只见过 4/5 的数据，用它们去上线不划算；
- CV 的任务是**估计泛化能力**，估计完了就丢掉 fold 模型，真正上线的是全量 fit 的那个。
- 所以 `blocked_mean` / `oof_accuracy` 应该理解成**部署模型精度的保守下限**（部署模型见过的数据更多，实际表现通常略高）。

---

## 4. 具体数字与读法

### 4.1 四只 RF 的 blocked CV 分数（当前训练结果）

| branch | N | segs | folds | mean | std | min | OOF acc |
|---|---|---|---|---|---|---|---|
| `active_motion` | 1411 | 15 | 5 | **0.9121** | 0.033 | 0.867 | 0.912 |
| `active_static` | 2614 | 11 | 5 | **0.9757** | 0.040 | 0.896 | 0.976 |
| `inactive_motion` | 2991 | 24 | 5 | **0.9400** | 0.042 | 0.866 | 0.940 |
| `inactive_static` | 2855 | 10 | 5 | **0.9843** | 0.013 | 0.969 | 0.984 |

### 4.2 字段怎么读

- **N** = 该 branch 的滑窗总数（每 100ms 数据 × 每 2 帧一个窗口 → 大致窗口数 ≈ 帧数 ÷ 2）
- **segs** = 该 branch 里 label-segment 的数量。越多说明训练/评测信号越丰富，5 折的随机性越小。
- **folds** = CV 折数 = `BLOCKED_N_FOLDS` = 5
- **mean** = 5 折 test accuracy 的平均 —— ★ **主要参考指标**
- **std** = 折间标准差，`< 0.05` 通常算稳定；`inactive_static=0.013` 说明各个时间段都很一致
- **min** = 最差那一折（**最坏情况**指标），上线时需要考虑这个数字
- **OOF acc** = 每个窗口被留出时那一次预测的总准确率（按窗口加权），和 mean 接近

### 4.3 混淆矩阵怎么看

每个 `rf_*.joblib` 的 `bundle['metrics']['confusion_oof']` 里存了一张 OOF 聚合混淆矩阵。以 `inactive_static` 为例：

```
                    STANDING_LEFT_LEAN   STANDING_RIGHT_LEAN   STANDING_UPRIGHT
STANDING_LEFT_LEAN                1137                    14                  5    n=1156
STANDING_RIGHT_LEAN                  8                   827                 16    n= 851
STANDING_UPRIGHT                     0                     2                846    n= 848
```

**行 = 真实类，列 = 预测类**，对角线是预测对的窗口数。读法：

- `STANDING_UPRIGHT` 真实 848 个，其中 846 预测对、2 预测成 RIGHT_LEAN、0 预测成 LEFT_LEAN → **recall = 846/848 = 99.76%**
- `STANDING_LEFT_LEAN` 这一列（= 所有被预测成 LEFT_LEAN 的窗口）共 1137+8+0=1145 个，其中真的是 LEFT_LEAN 的 1137 个 → **precision = 1137/1145 = 99.30%**
- 行里数字分散到多列 = 该类经常被误判
- 某列几乎是 0 = 模型几乎不会预测到这一类（值得警惕）

训练脚本也会把这张矩阵保存成 `confusion_matrices/confusion_<branch>_blocked_oof.png`（带颜色）。

---

## 5. 如何在本地复现 / 查看

### 5.1 重新训练 + 打印所有指标

```bash
# 默认就是 blocked 模式
python ml_train_branch_rfs.py

# 想对比 LOFO
python ml_train_branch_rfs.py --eval-mode lofo

# 想对比"旧版随机 shuffle 60/20/20"
python ml_train_branch_rfs.py --eval-mode random
```

每只 RF 训练过程中会打印：

- `[BLOCKED-CV] n_segments=... n_folds=5 purge=5 windows`
- 每折准确率 `mean / std / min / max`
- OOF accuracy
- **ASCII 混淆矩阵** + **classification_report**
- 保存 `confusion_matrices/confusion_<branch>_blocked_oof.png`
- `saved model: rf_<branch>.joblib (trained on all N windows)`

### 5.2 不重训，只从 joblib 里读回分数

```python
import joblib
bundle = joblib.load('rf_inactive_static.joblib')
m = bundle['metrics']

print('evaluation mode :', m['evaluation'])              # 'blocked_intra_file'
print('N windows       :', m['n_samples'])
print('5-fold mean/std :', m['blocked_mean'], m['blocked_std'])
print('OOF accuracy    :', m['oof_accuracy'])
print('classes         :', m['classes'])
print('confusion_oof   :', m['confusion_oof'])           # 2D list，行列都按 classes 顺序
```

在 `quickstart_guide.ipynb` 的 §4.2 和 `smart_sock_pipeline.ipynb` 的 §6.1.1 都有现成的展示 cell。

### 5.3 验证每一折的 test 都覆盖全部类（debug 用）

```python
import numpy as np
import ml_train_branch_rfs as mtb
from ml_activity_features import load_csv_files
import personal_calibration as pc

raw, labels, _, _ = load_csv_files(mtb.DATA_DIR, labeled_only=True, raw_adc=True)
calib = pc.OfflineAutoCalibrator().fit(raw, labels, subject='test')
per_file = mtb._load_csv_per_file(mtb.DATA_DIR)
X, y, b, g, s = mtb._windows_for_branch_per_file(per_file, calibration=calib)

for br in sorted(set(b)):
    mask = np.array([x == br for x in b])
    fold_ids = mtb._assign_blocked_folds(s[mask], n_folds=5)
    cls = set(y[mask])
    for k in range(5):
        test_cls = set(y[mask][fold_ids == k])
        missing = cls - test_cls
        print(br, 'fold', k, 'missing:', sorted(missing) if missing else '(all present)')
```

---

## 6. 三种评测模式并排对比

| 维度 | random shuffle | LOFO | **blocked intra-file**（默认）|
|------|---------------|------|-------------------------------|
| test 集里每类都存在？ | ✅ | ❌ 很多 fold 缺类 | ✅ |
| 相邻帧不跨 train/test？ | ❌ 严重泄漏 | ✅ 整文件隔离 | ✅ 块内连续 + purge gap |
| 每窗口恰被预测 1 次？ | ❌ 只有 test 集那部分 | ✅ | ✅ |
| 分数虚高程度 | 极高（会到 99%） | 偏低（方差大） | **真实** |
| fold 数 | 单次 split | = 文件数（如 33） | `BLOCKED_N_FOLDS = 5` |
| 开销 | 最低 | 最高 | 中 |
| 适合场景 | 快速 debug | 严格跨受试者评测（需每受试者都录多类） | **本项目默认** |

---

## 7. 常见 FAQ

### Q1: 为什么 fold 数固定 5？

段切太细（K=10+）每段每块就太短，模型训练时 label-segment 之间的 class balance 容易抖动；太粗（K=3）test 块占比 33%，train 数据就少了。K=5 = 每 fold test 占 20% 窗口，和传统 5 折直觉一致，且总耗时可接受。想改：`ml_train_branch_rfs.py` 顶部 `BLOCKED_N_FOLDS`。

### Q2: 为什么 purge = 5？

= `WINDOW_SIZE / WINDOW_STEP` = 10 / 2 = **一个完整窗口的跨度**。这保证 train / test 窗口没有任何一个共同原始帧。想改：`BLOCKED_PURGE`。

### Q3: 为什么不报告"validation + test"两个数字，而是只有一个 CV mean？

传统 "60/20/20 random shuffle" 里的 val 和 test 本质是"两个互相独立的 random holdout"，在时间序列场景下这两个分数都同时被泄漏。**blocked CV 给的 5 折分数（mean + std）就是同一个指标更稳健的估计**，同时 `min` 对应"最坏那个 holdout"、`OOF` 对应"全量 holdout 聚合"，信息更完整。

如果确实需要单独一个"test-only"数字：可以把 `blocked_fold_accuracies` 的**最后一折当 test**、前 4 折当 CV mean/std，语义等价于"4 折 CV + 1 固定 test"。但因为 5 个 fold 对称，没必要强行这么拆。

### Q4: 部署模型用了全量 fit，这不是过拟合吗？

**不是**。过拟合的定义是"在未见数据上表现差"。CV 已经在 5 个没用过的 test 块上独立测过了同一种 pipeline / 超参，得到的 mean/std 就是**部署模型在新数据上表现的无偏估计**。用全量数据 fit 只是把 "每折 RF 各自丢了 1/5 训练数据" 的损失补回来。

更严谨的说法：blocked CV 估计的是 **pipeline 的泛化能力**（特征工程 + 超参 + 样本量 ≈ 9871 时的能力），部署模型用同一条 pipeline 在 full data 上 fit，所以它的泛化能力**不会低于** CV 给出的均值（通常略高）。

### Q5: 想引入新用户 / 新受试者的数据 how？

- 用 UI 的在线标定（`foot_pressure_monitor.py` 里的 Calibration 选项卡）给新用户做个人量程校准；
- 如果想让模型也"学会"这个新用户，把他采的 CSV 放进 `saving_data/` 重跑训练即可。blocked CV 会自动把新数据参与进来。

---

## 附录 · 代码索引

- 窗口生成 + segment 标注：`ml_train_branch_rfs.py::_windows_for_branch_per_file`
- Fold 分配（连续块）：`ml_train_branch_rfs.py::_assign_blocked_folds`
- 带 purge 的 CV iterator：`ml_train_branch_rfs.py::BlockedIntraFileCV`
- 单 branch 训练 + 评测：`ml_train_branch_rfs.py::_train_one_branch`
- CLI 入口 + 总表打印：`ml_train_branch_rfs.py::main`
- 全局归一化（所有 CSV 合并后计算）：`personal_calibration.py::OfflineAutoCalibrator.fit`
- 特征管道冻结 EWMA：`adaptive_preprocessing.py::AdaptiveSensorPreprocessor` (构造参数 `seed`)
