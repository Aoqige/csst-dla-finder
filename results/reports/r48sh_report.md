# R48-SH — Unified VAL Single-Checkpoint Score Hunt

## 1. 实验改了什么

在 unified TRAIN 上从头训练 8 个新 seed（46–53），每个 20 epoch，EMA beta=0.999；
对 8 个 EMA endpoint（6/8/10/12/14/16/18/20）各跑一次 R38 式 matched-WLS 读出，共 64 个候选；
只在 unified VAL 上按 Final 取最大。TEST 未评分、未调参、未用于选择。

## 2. 结果

- 候选总数（预期 / 实际评分）：64 / 64
- 缺失候选数：0
- 全部 gate 通过的候选数：64
- 全部候选完成且通过完整性核查：True
- 最高单模型 VAL Final：0.687569327685616
- 最低 / 中位：0.6630952592555015 / 0.678991362012193
- 超过 0.687357 的候选数：1
- 超过 0.690000 / 0.695000 / 0.700000：0 / 0 / 0
- 距 0.700000：0.012430672314383995
- 最高点相对历史 0.687357：0.00021232768561596504

最高点：seed **51**，EMA epoch **8**，
Final 0.687569327685616 = 0.6×0.6712027088378448 + 0.4×0.7121192559572727；
checkpoint `/home/heruihua/csst_dla_runs/20261008_r48sh/seed51/ema_ep8.pt`（md5 `889e42c20ee1be032fad4a084ae3f195`）。

### 各 EMA epoch 的最高分

| EMA epoch | 候选数 | 该 epoch 最高 Final |
|---:|---:|---:|
| 6 | 8 | 0.6870428480401215 |
| 8 | 8 | 0.687569327685616 |
| 10 | 8 | 0.686900647846922 |
| 12 | 8 | 0.6843506562417734 |
| 14 | 8 | 0.6805342188842552 |
| 16 | 8 | 0.6781790541626853 |
| 18 | 8 | 0.6758567000681748 |
| 20 | 8 | 0.6735331175123487 |

### 历史只读参考（不进入 64 候选预算）

| seed | 来源 | VAL Final |
|---|---|---:|
| 42 | R38 matched-WLS | 0.6825514434232297 |
| 43 | R38 matched-WLS | 0.6831270401484937 |
| 44 | R38 matched-WLS | 0.6829074029348012 |
| 45 | R39 matched-WLS | 0.6870866739276652 |
| 21 | count_loss_weight=0.10 best | 0.687357 |

## 3. 实验有效性

- 评分沿用 R42/R47b 已核实的官方 float32 口径；未在评分前把物理预测字段转 float64。
- 每个候选的完整性 gate（两塔权重未变、WLS 只改 lognhi_delta、几何与 matched 集合不变、
  checkpoint 可重载、gate 全过）由 R38 Stage-B 脚本自身记录在 `gates` 字段。
- 未做 threshold sweep、count-bias search、LR/EMA sweep、WLS ridge/weight search、
  seed 扩展、epoch 追加、参数 ensemble、split 更换；未做 TEST 评分。
- 最高分属 repeated-development selection（unified VAL），不构成无偏泛化成绩。

## 4. 尚缺的数据

- 本表只覆盖 8 seed × 8 EMA endpoint；未覆盖其它 epoch 或其它 seed。
- 未评估所选最高点在其它 split 上的行为（本轮禁止 TEST 评分）。
- 未做多候选一致性/稳定性统计（本轮目标为 single-checkpoint maximum）。

