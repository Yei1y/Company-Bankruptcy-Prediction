# 结果记录与数值核对日志

本文件记录**每次脚本运行后的数值结果**，以及 README 与论文中数字的**来源与口径差异**。
规则：代码只负责计算并落盘，解读与结论写在这里；所有数值都指向仓库内的结果文件。

- 记录时间：2026 年（README 重写与结果核验）
- 本次工作**没有重新训练任何模型**（SIS / SCAD / LightGBM / SMOTE 均未重跑），只读取已落盘的结果对象重新计算
- 唯一新执行的脚本：[`scripts/99_export_result_tables.R`](scripts/99_export_result_tables.R)，只读 `data/*.rds` 与 `data/*.RData`，把数值导出为 `results/tables/*.csv`
- 本机实测耗时：**58 秒**（主要开销是两次读取 1.1GB 与 450MB 的超高维矩阵）

---

## 1. 数据概览（来源：`data/data.csv`、`data/01_*.rds`）

导出文件：[`results/tables/data_overview.csv`](results/tables/data_overview.csv)

| 项目 | 数值 |
|------|------|
| 原始样本数 × 列数 | 6,819 × 96 |
| 目标变量 | `Bankrupt?`（位于首列） |
| 常数列（剔除） | `Net Income Flag`（方差为 0） |
| 参与建模特征数 | 94 |
| 训练集 / 测试集 | 4,773 / 2,046（7:3 划分，seed = 1111） |
| 训练集破产 / 健康 | 144 / 4,629（破产占比 3.017%） |
| 测试集破产 / 健康 | 76 / 1,970（破产占比 3.715%） |
| 全样本破产占比 | 3.2263%（220 / 6,819） |
| 缺失值 | 0 |

**判断**：类别不平衡比约 1:30（训练集 144:4,629 = 1:32.15），本文使用代价敏感权重而非重采样，权重即取 `n0/n1 = 32.1458`。

> ⚠️ **与旧 README 的差异**：此前仓库 README 写"327 破产，6492 非破产"，与 `data.csv` 实际不符（正确为 220 / 6,599）。本次已按实际数据改正。

## 2. 超高维特征空间构成（来源：`data/02_Train_Ultrahigh.rds` 列名）

导出文件：[`results/tables/expansion_summary.csv`](results/tables/expansion_summary.csv)、[`block_membership.csv`](results/tables/block_membership.csv)、[`spline_expansion_note.csv`](results/tables/spline_expansion_note.csv)

训练集矩阵维度实测为 **4,773 × 31,616**，构成拆解如下：

| 组成 | 列数 | 倍数 |
|------|------|------|
| 原始特征（线性主效应） | 94 | 1.00× |
| 单变量样条基 / 降级线性项 | 280 | 2.98× |
| 块间张量积交互项 | 31,242 | 332.36× |
| **合计** | **31,616** | **336.34×** |

- 样条列数 280 = 93 × 3 + 1：`Liability-Assets Flag` 因取值极端集中导致节点重合，被捕获异常后**降级为线性项**（`Lev.Liability.Assets.Flag_linear`），与论文正文的描述一致。
- 模块规模（由单变量样条列名的模块前缀反解，比脚本内的索引列表更可靠）：Prof 20、Lev 21、Cash 21、Eff 17、Grow 15，合计 **94**。
- 与 `notebooks/02` 中索引列表定义的模块**逐列一致**（5 个模块全部 0 处不一致，见 [`block_membership_check.csv`](results/tables/block_membership_check.csv)）。

## 3. 单变量信号强度（来源：`data/01_Train_Data.rds`）

导出文件：[`results/tables/univariate_signal.csv`](results/tables/univariate_signal.csv)、[`top10_univariate_features.csv`](results/tables/top10_univariate_features.csv)

| 项目 | 数值 |
|------|------|
| 特征数 | 94 |
| \|r\| 中位数 | 0.0615 |
| \|r\| 均值 | 0.0812 |
| \|r\| 最大值 | 0.3084（`Net.Income.to.Total.Assets`） |
| \|r\| ≥ 0.3 | 1 个 |
| \|r\| ≥ 0.2 | 8 个 |
| \|r\| < 0.1 | 65 个 |

**判断**：单变量点二列相关整体极弱，靠"选相关最强的若干变量"无法刻画破产机制；这为显式构造非线性与交互特征提供了依据。

## 4. 模块级相关结构（来源：`data/01_Train_Data.rds`）

导出文件：[`results/tables/block_correlation_matrix.csv`](results/tables/block_correlation_matrix.csv)

平均绝对相关（对角为模块内，非对角为模块间）：

| | Prof | Lev | Cash | Eff | Grow |
|---|---|---|---|---|---|
| **Prof** | 0.2370 | 0.0647 | 0.0974 | 0.0706 | 0.1552 |
| **Lev** | 0.0647 | 0.1492 | 0.0708 | 0.0583 | 0.0447 |
| **Cash** | 0.0974 | 0.0708 | 0.1145 | 0.0455 | 0.0642 |
| **Eff** | 0.0706 | 0.0583 | 0.0455 | 0.0525 | 0.0434 |
| **Grow** | 0.1552 | 0.0447 | 0.0642 | 0.0434 | 0.1681 |

- 模块内平均 **0.1443**，模块间平均 **0.0715**（约 2.0 倍）。模块内相关最高的是盈利（0.2370）与成长（0.1681），最低的是效率（0.0525）。
- 模块间相关最强的是盈利 × 成长（0.1552），其余多在 0.04–0.10 之间。
- **判断**：信息冗余集中在模块内部，模块间相对独立——这正是"模块内用样条刻画非线性、模块间做张量积交叉"这一设计的经验支撑。

## 5. GCor-SIS-REDS-MDS 筛选（来源：`data/03_SIS_Results_Full.rds`）

导出文件：[`results/tables/sis_selection_summary.csv`](results/tables/sis_selection_summary.csv)、[`sis_frequency_bins.csv`](results/tables/sis_frequency_bins.csv)

参数（`scripts/03_SIS.R`）：α = 0.1、B = 20 次拆分、单次拆分内 $D_1:D_2$ = 2:1、τ = 0.5。

| 阶段 | 特征数 | 占 31,616 的比例 |
|------|--------|------------------|
| 至少入选 1 次 | 26,552 | 84.0% |
| 频率 ≥ 0.5（脚本 τ） | 12,043 | 38.1% |
| 频率 > 0.7（SCAD 实际口径） | **5,218** | 16.5% |
| 频率 = 1.0 | **0** | — |

频率分布（每 0.05 一档）：`[0,0.05]` 2,144；(0.05,0.1] 1,088；(0.1,0.15] 1,260；(0.15,0.2] 1,253；(0.2,0.25] 1,312；(0.25,0.3] 1,338；(0.3,0.35] 1,887；(0.35,0.4] 2,261；(0.4,0.45] 1,966；(0.45,0.5] 1,087；(0.5,0.55] 1,132；(0.55,0.6] 1,619；(0.6,0.65] 1,069；(0.65,0.7] 1,918；**(0.7,0.75] 1,866**；(0.75,0.8] 682；(0.8,0.85] 667；(0.85,0.9] 679；**(0.9,0.95] 1,324**；(0.95,1] 0。最高频率 0.95，中位数 0.4500。

**判断**：论文称"选择频率多处于 (0.70,0.75] 与 (0.90,0.95] 两个区间"——与落盘结果一致。**没有任何特征在 20 次拆分中被 100% 选中**，这是使用多重拆分（MDS）而非单次拆分的直接理由。

> ⚠️ **口径说明**：论文正文所报"筛选出 5,218 个特征"对应的是 **频率 > 0.7** 的口径（`03_SCAD.R` 第 18 行的 `which(selection_freq > 0.7)`），不是脚本中 τ = 0.5 的口径（12,043）。两个数字都正确，但含义不同，引用时需注明。

## 6. SCAD 模型（来源：`data/03_SCAD_Model.rds`、`data/03_SCAD_Selected_Features.rds`）

导出文件：[`results/tables/scad_model_summary.csv`](results/tables/scad_model_summary.csv)、[`scad_selected_features.csv`](results/tables/scad_selected_features.csv)

| 项目 | 数值 |
|------|------|
| CV 折数 | 10（`createFolds`，seed = 1） |
| γ（SCAD 参数 a） | 3.7 |
| λ 网格数 | 150 |
| λ.min | 0.004611 |
| 类别权重 n0/n1 | 32.1458 |
| **非零系数特征数** | **16** |
| 其中交互项 | 15 |
| 其中含杠杆模块 | 13 |
| 含 ns2 或 ns3 基函数的交互项 | 11 / 15 |

按 |SCAD 系数| 降序（与论文表 3 的 F 编号一致）：F01 (+65.60)、F02 (−25.83)、F03 (−9.85)、F04 (−7.85)、F05 (−7.42)、F06 (−6.47)、F07 (−5.05)、F08 (+3.50)、F09 (−3.04)、F10 (−2.61)、F11 (+1.34)、F12 (−1.02)、F13 (+0.27)、F14 (−0.0143，唯一纯线性项)、F15 (0.00)、F16 (0.00)。

模块构成：杠杆 × 效率 6、杠杆 × 成长 4、盈利 × 杠杆 2、杠杆 × 现金流 1、盈利 × 效率 1、盈利 × 现金流 1。

**判断**：16 个幸存特征中 15 个是交互项，13 个含杠杆模块，说明模型把"杠杆与其他维度的非线性交叉"识别为破产的主要驱动结构，与论文"以杠杆为枢纽的多重财务恶化非线性共振"的表述方向一致。

## 7. 测试集评估（来源：`data/03_Evaluation_Metrics.RData`）

导出文件：[`results/tables/model_metrics_scad.csv`](results/tables/model_metrics_scad.csv)、[`confusion_matrix_scad.csv`](results/tables/confusion_matrix_scad.csv)

| 指标 | 数值 |
|------|------|
| ROC-AUC | 0.9380 |
| PR-AUC | 0.4222 |
| KS 统计量 | 0.7638 |
| 最优阈值（Youden） | 0.0317 |
| 精确率 (PPV) | 0.2086 |
| 召回率 | 0.8947 |
| F1 / F2 | 0.3383 / 0.5397 |

混淆矩阵（阈值 0.0317）：TN = 1,712、FP = 258、FN = **8**、TP = **68**。

**判断**：模型在 76 家破产企业中识别出 68 家（召回 89.5%），代价是 258 家健康企业被误判；PR-AUC 仅 0.4222 是 3.23% 先验下的必然结果，而非模型缺陷。

## 8. 后选择推断——方案 A：全量重拟合（来源：`data/03_SCAD_Inference_Results.rds`）

导出文件：[`results/tables/post_selection_inference_full.csv`](results/tables/post_selection_inference_full.csv)、[`post_selection_inference_summary.csv`](results/tables/post_selection_inference_summary.csv)

16 个 SCAD 变量中剔除 5 个（F07、F10、F13、F15、F16：共线别名化或标准误 > 10），在 11 个变量上重拟合并做 B = 200 次 Bootstrap：

| 项目 | 数值 |
|------|------|
| 显著变量数（α = 0.05，不含截距） | **8 / 11** |
| 不显著 | F06 (0.146)、F11 (0.170)、F14 (0.054) |
| Bootstrap / Wald 区间平均宽度比 | **3.8639** |

**判断**：Bootstrap 区间平均比 Wald 区间宽约 2.9 倍（宽度比 3.86），两者存在系统性差异。交互特征空间的结构性共线性使 Fisher 信息矩阵趋于病态，Wald 区间偏窄、易过度自信；Bootstrap 不依赖信息矩阵可逆性，应作为主要参考。方案 A 本身含有选择偏差（选择与推断使用同一份数据），其 p 值只能视为上界。

## 9. 后选择推断——方案 B：拆分样本（来源：`data/03_SCAD_Split_Inference.rds`、`data/03_Split_Evaluation_Metrics.RData`）

导出文件：[`results/tables/post_selection_inference_split.csv`](results/tables/post_selection_inference_split.csv)、[`split_inference_summary.csv`](results/tables/split_inference_summary.csv)

| 项目 | 数值 |
|------|------|
| 拆分种子 | 2024 |
| 子集 A（选择，加权） | n = 2,386 |
| 子集 B（推断，无权重） | n = 2,387 |
| 子集 A 的 SCAD 选中特征数 | **2**（λ.min = 0.014435） |
| 子集 B 的推断保留变量数 | 2，且均显著 |

| 变量 | 特征 | 系数 | 标准误 | p 值 |
|------|------|------|--------|------|
| F01 | 盈利 × 杠杆：Net Income to Total Assets (ns2) × Net worth/Assets (ns2) | 61.315 | 5.510 | <0.001 |
| F06 | 现金流 × 效率：Quick Ratio (ns3) × Inventory and accounts receivable/Net value (ns1) | 15.288 | 5.613 | 0.006 |

**判断**：**这是本次核验中最重要的一条结果**。方案 A 的"8 / 11 显著"在方案 B 中无法复现：仅仅换掉一半样本，SCAD 的选择本身就收缩到 2 个变量。方向上与选择偏差的预期完全一致，说明在 3.23% 正类先验与高度共线的交互特征空间下，**变量选择的稳定性是结论最大的不确定性来源**。论文正文目前只报告了方案 A，尚未纳入这一对照。

## 10. 经济效益与决策阈值（来源：`data/03_SCAD_Model.rds` + `data/02_Test_Ultrahigh.rds` 重新打分）

导出文件：[`results/tables/economic_scenarios.csv`](results/tables/economic_scenarios.csv)、[`test_probabilities.csv`](results/tables/test_probabilities.csv)

用后验比例做反向概率校准 $\tilde p = \hat p_w / (\hat p_w + 32(1-\hat p_w))$ 还原物理违约概率，再按 $p^* = r/(r+\mathrm{LGD})$ 决策：

| 情景 | 最优阈值 | 批准放贷 | 其中违约 | 净利润 |
|------|----------|----------|----------|--------|
| 信用贷：利率 5%、LGD = 100% | 0.0476 | 2,025 | 63 | 3,510 万元 |
| 抵押贷：利率 6%、LGD = 40% | 0.1304 | 2,035 | 71 | 8,944 万元 |

- 两个情景的净利润与论文摘要所报"3,510 万元 / 8,944 万元"**完全一致**，说明论文数字可由落盘模型复算（本次即用已保存的 SCAD 模型重新打分，未重新训练）。
- 由 Youden 阈值 0.0317 反解 $p^*=r/(r+\mathrm{LGD})$，可得漏报/误杀的代价比 **30.5 : 1**，与论文"漏报一家破产企业的代价是误杀一家健康企业的 30.5 倍"一致。

> ⚠️ **取整问题**：`scripts/04_economic.R` 第 38 行把校准常数**硬编码为 32**，而 SCAD 的实际权重比为 4,629/144 = **32.1458**。差异对阈值与利润的影响极小，但严格来说，"反向校准 = King & Zeng 截距修正"的等价性只在权重比恰为整数时成立。

---

## 11. 与论文的口径差异清单（供修改论文时逐条核对）

以下差异均由落盘结果反查得到，**不是对论文结论的否定，而是编号与表述层面的不一致**：

| 编号 | 论文表述 | 落盘结果 / 实际数据 | 影响 |
|------|----------|---------------------|------|
| D1 | 表 4 中系数 −2.86 的行标为 `F13` | 按论文表 3 的映射与该行数值，它对应 `F12`（盈利 × 现金流）；结果文件用 `F12` | 表格标签错位一处 |
| D2 | 正文称"剔除 F07、F10、**F12**、F15 及 F16" | 结果文件实际剔除的是 F07、F10、**F13**、F15、F16 | 与 D1 同源 |
| D3 | 正文称"显著变量中有 **7** 项涉及杠杆维度" | 按论文表 3 的映射统计为 **6** 项（F01–F05、F09） | 差异源于 D1：把盈利 × 现金流那一行误当作杠杆 × 效率后才会得到 7 |
| D4 | 正文称"F05（流动负债/权益 ns1 × 净值周转率 **ns2**）" | 结果文件为 `Net.Worth.Turnover.Rate..times._ns1`，即 **ns1** | 表述笔误 |
| D5 | 摘要/正文称 SIS"筛选出 5,218 个特征" | 5,218 = 频率 > 0.7 的口径；脚本 τ = 0.5 的口径为 12,043 | 口径需注明 |
| D6 | （旧 README）"327 破产，6492 非破产" | 实际 220 / 6,599 | 旧 README 数字错误，已改正 |

## 12. 未复核的数字（明确标注）

以下数字**只来自论文，仓库中没有对应的结果文件，本次未复核**：

| 数字 | 论文出处 | 缺失的结果文件 |
|------|----------|----------------|
| LightGBM：AUC 0.9404、PR-AUC 0.4561、KS 0.7663、F1 0.3255、F2 0.5291；有效特征 91 个 | 表 5、第 6 节 | `data/04_LightGBM_Evaluation.RData`（`03b_LightGBM.R` 会生成） |
| SMOTE + 逻辑回归：AUC 0.8207、PR-AUC 0.2246、KS 0.6415、F1 0.4060、F2 0.5466 | 表 5 | `data/04_LogisticRegression_Evaluation.RData`（`03c_SMOTE_LG.R` 会生成） |

重跑 `scripts/03b_LightGBM.R` 与 `scripts/03c_SMOTE_LG.R` 即可补齐并复核（两者在 94 维原始特征上训练，计算量远小于 SIS / SCAD 阶段），本次按"不重跑代码"的要求未执行。

## 13. 其他需要记录的工程问题

| 问题 | 说明 | 影响 |
|------|------|------|
| SIS 阶段无随机种子 | `03_SIS.R` 未调用 `set.seed()`，而 MDS 依赖随机拆分 | SIS 结果无法逐位复现（本次仅核对已落盘的那一次运行） |
| `data/03_SCAD_Selected_Features.rds` 无生成脚本 | 现有脚本均未写出该文件；其内容与 `cv_scad` 的 16 个非零系数一致 | 已由 `99_export_result_tables.R` 重新导出为 CSV 以供核对 |
| `03_SCAD_Split.R` 与产出版本不一致 | 落盘结果的短 ID 为 F01 / F06，而当前脚本按顺序编号会得到 F01 / F02 | 重跑脚本得到的标签可能与落盘结果不同，特征本体一致 |
| `scripts/00_Data_Analyze.R` 的模块索引 | 该脚本在含常数列的 95 维矩阵上使用索引，杠杆模块把常数 `Net Income Flag` 计入、漏掉 `Liability-Assets Flag` | 只影响探索性图表；建模用的 94 维模块划分已核对为 20/21/21/17/15 且与列名一致 |
| 4 个大文件未入库 | `02_*_Ultrahigh.rds`（1.1GB / 450MB）、`03_Evaluation_Metrics.RData`（208MB）、`03_SCAD_Inference_Results.rds`（198MB）超过 GitHub 单文件 100MB 限制 | 复现者需重跑 notebook 02 与相关脚本；README 已说明 |
| CSV 编码 | R 在 Windows 下默认按本地编码写出，本次统一使用 `fileEncoding = "UTF-8"` | 20 个 CSV 已全部验证为合法 UTF-8，GitHub 上不会乱码 |