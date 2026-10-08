# 脚本功能：从已落盘的结果文件中提取数值，导出为 results/tables/*.csv
# 只读取 01/02/03 阶段保存的 rds 与 RData，不重新训练任何模型
# 运行方式：从 scripts/ 目录执行 Rscript 99_export_result_tables.R

library(ncvreg)
library(pROC)

out_dir <- "../results/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 统一以 UTF-8 写出，避免 Windows 本地编码导致中文在 GitHub 上乱码
write_utf8 <- function(x, name, ...) write.csv(x, file.path(out_dir, name), ..., fileEncoding = "UTF-8")

# 读取原始数据与划分后的数据集
raw <- read.csv("../data/data.csv", check.names = FALSE)
train_data <- readRDS("../data/01_Train_Data.rds")
test_data  <- readRDS("../data/01_Test_Data.rds")
Y_train <- as.numeric(as.character(train_data$Y))
Y_test  <- as.numeric(as.character(test_data$Y))

# 数据规模与类别分布
constant_cols <- names(raw)[sapply(raw, function(x) length(unique(x)) == 1)]
overview <- data.frame(
  item = c("原始样本数", "原始列数", "目标变量", "常数列(剔除)",
           "参与建模特征数", "训练集样本数", "训练集破产数", "训练集健康数",
           "测试集样本数", "测试集破产数", "测试集健康数",
           "全样本破产占比", "训练集破产占比", "测试集破产占比", "缺失值数量"),
  value = c(nrow(raw), ncol(raw), names(raw)[1], paste(constant_cols, collapse = ";"),
            ncol(train_data$X), length(Y_train), sum(Y_train == 1), sum(Y_train == 0),
            length(Y_test), sum(Y_test == 1), sum(Y_test == 0),
            round(mean(raw[[1]]) * 100, 4), round(mean(Y_train) * 100, 4),
            round(mean(Y_test) * 100, 4), sum(is.na(raw)) + sum(is.nan(as.matrix(raw))))
)
write_utf8(overview, "data_overview.csv", row.names = FALSE)
print(overview, row.names = FALSE)

# 超高维特征空间构成：原始项 / 单变量样条基 / 块间张量积交互项
ultra_names <- colnames(readRDS("../data/02_Train_Ultrahigh.rds")$X)
is_interaction <- grepl("_X_", ultra_names, fixed = TRUE)
is_spline <- !is_interaction & grepl("_(ns[0-9]+|linear)$", ultra_names)
expansion <- data.frame(
  item = c("原始特征(线性主效应)", "单变量样条基/降级线性项", "块间张量积交互项", "合计"),
  n_cols = c(sum(!is_interaction & !is_spline), sum(is_spline), sum(is_interaction), length(ultra_names))
)
expansion$ratio_to_raw <- round(expansion$n_cols / ncol(train_data$X), 2)
write_utf8(expansion, "expansion_summary.csv", row.names = FALSE)
cat("\n[特征空间构成]\n"); print(expansion, row.names = FALSE)

# SIS 阶段的选择频率分布
sis <- readRDS("../data/03_SIS_Results_Full.rds")
freq <- sis$selection_freq
bins <- seq(0, 1, by = 0.05)
freq_tab <- as.data.frame(table(cut(freq[freq > 0], breaks = bins, include.lowest = TRUE)))
colnames(freq_tab) <- c("freq_interval", "n_features")
sis_summary <- data.frame(
  item = c("被选中至少一次的特征数", "频率>=0.5(脚本tau)", "频率>0.7(SCAD阶段阈值)",
           "频率=1.0的特征数", "最高频率", "频率中位数(>0)", "频率均值(>0)"),
  value = c(sum(freq > 0), sum(freq >= 0.5), sum(freq > 0.7), sum(freq == 1),
            max(freq), median(freq[freq > 0]), round(mean(freq[freq > 0]), 4))
)
write_utf8(sis_summary, "sis_selection_summary.csv", row.names = FALSE)
write_utf8(freq_tab, "sis_frequency_bins.csv", row.names = FALSE)
cat("\n[SIS 汇总]\n"); print(sis_summary, row.names = FALSE)
cat("\n[SIS 频率分布]\n"); print(freq_tab, row.names = FALSE)

# SCAD 模型与非零系数特征
cv_scad <- readRDS("../data/03_SCAD_Model.rds")
coefs <- coef(cv_scad)
active <- coefs[-1][coefs[-1] != 0]
active <- active[order(abs(active), decreasing = TRUE)]
sel <- data.frame(
  feature_id = paste0("F", sprintf("%02d", seq_along(active))),
  feature = names(active),
  coefficient = round(as.numeric(active), 4)
)
sel$is_interaction <- grepl("_X_", sel$feature, fixed = TRUE)
# 交互项列名形如 <块1>_<块2>.<块1>_<块2>_<变量A>_ns#_X_<变量B>_ns#，块标签取最前两个下划线分隔的记号
tokens <- strsplit(sel$feature, "_", fixed = TRUE)
block_a <- ifelse(sel$is_interaction, vapply(tokens, `[`, character(1), 1), NA)
block_b <- ifelse(sel$is_interaction, vapply(tokens, `[`, character(1), 2), NA)
block_b <- sub("[.].*$", "", block_b)
sel$blocks <- ifelse(sel$is_interaction, paste(block_a, block_b, sep = " x "), "无(纯线性主效应)")
sel$involves_leverage <- sel$is_interaction & (block_a == "Lev" | block_b == "Lev")
sel$spline_orders <- vapply(regmatches(sel$feature, gregexpr("ns[0-9]+|linear", sel$feature)),
                            function(v) paste(v, collapse = "+"), character(1))
write_utf8(sel, "scad_selected_features.csv", row.names = FALSE)
cat("\n[SCAD 选中特征]\n"); print(sel, row.names = FALSE)

scad_summary <- data.frame(
  item = c("CV 折数", "gamma", "lambda 网格数", "lambda.min", "类别权重(n0/n1)",
           "非零系数特征数", "其中交互项", "其中含杠杆模块", "含ns2或ns3基函数的交互项",
           "训练集样本数"),
  value = c(10, cv_scad$fit$gamma, length(cv_scad$lambda), round(cv_scad$lambda.min, 6),
            round(sum(Y_train == 0) / sum(Y_train == 1), 4), length(active),
            sum(sel$is_interaction), sum(sel$involves_leverage),
            sum(sel$is_interaction & grepl("ns[2-9]", sel$spline_orders)),
            length(Y_train))
)
write_utf8(scad_summary, "scad_model_summary.csv", row.names = FALSE)
cat("\n[SCAD 模型汇总]\n"); print(scad_summary, row.names = FALSE)

# 测试集评估指标与混淆矩阵
load("../data/03_Evaluation_Metrics.RData")
metrics <- eval_metrics_df
write_utf8(metrics, "model_metrics_scad.csv", row.names = FALSE)
cm <- as.data.frame.matrix(conf_matrix$table)
cm$prediction <- rownames(cm)
write_utf8(cm, "confusion_matrix_scad.csv", row.names = FALSE)
cat("\n[主模型评估]\n"); print(metrics, row.names = FALSE); print(conf_matrix$table)

# 后选择推断结果（全量重拟合）
inf <- readRDS("../data/03_SCAD_Inference_Results.rds")
write_utf8(inf, "post_selection_inference_full.csv", row.names = FALSE)
width_ratio <- mean(inf$Bootstrap_CI_Upper - inf$Bootstrap_CI_Lower, na.rm = TRUE) /
  mean(inf$Analytic_CI_Upper - inf$Analytic_CI_Lower, na.rm = TRUE)
inf_summary <- data.frame(
  item = c("SCAD 初始选中", "剔除不稳定变量后", "显著变量数(alpha=0.05,不含截距)",
           "Bootstrap 次数", "Bootstrap/Wald 区间平均宽度比", "剔除变量的短ID"),
  value = c(length(readRDS("../data/03_SCAD_Selected_Features.rds")), nrow(inf) - 1,
            sum(inf$P_Value[inf$Variable != "(Intercept)"] < 0.05), 200,
            round(width_ratio, 4),
            paste(setdiff(sprintf("F%02d", 1:16), inf$Variable), collapse = ","))
)
write_utf8(inf_summary, "post_selection_inference_summary.csv", row.names = FALSE)
cat("\n[后选择推断汇总]\n"); print(inf_summary, row.names = FALSE)

# 拆分样本推断结果
inf_split <- readRDS("../data/03_SCAD_Split_Inference.rds")
write_utf8(inf_split, "post_selection_inference_split.csv", row.names = FALSE)
e2 <- new.env(); load("../data/03_Split_Evaluation_Metrics.RData", envir = e2)
split_summary <- data.frame(
  item = c("拆分种子", "子集A样本数(按脚本拆分规则)", "子集B样本数(按脚本拆分规则)",
           "子集A SCAD 选中特征数", "子集A lambda.min", "子集B 推断保留变量数",
           "显著变量数(alpha=0.05,不含截距)"),
  value = c(2024, floor(length(Y_train) / 2), length(Y_train) - floor(length(Y_train) / 2),
            length(e2$selected_vars), round(e2$cv_scad_A$lambda.min, 6), nrow(inf_split) - 1,
            sum(inf_split$P_Value[inf_split$Variable != "(Intercept)"] < 0.05))
)
write_utf8(split_summary, "split_inference_summary.csv", row.names = FALSE)
cat("\n[拆分样本推断]\n"); print(split_summary, row.names = FALSE); print(inf_split, row.names = FALSE)

# 经济效益：用已保存的 SCAD 模型对测试集重新打分并还原真实概率
test_ultra <- readRDS("../data/02_Test_Ultrahigh.rds")
idx <- which(sis$selection_freq > 0.7)
X_test_sis <- as.matrix(test_ultra$X)[, idx, drop = FALSE]
raw_coefs <- coef(cv_scad, lambda = cv_scad$lambda.min)
linear_predictor <- raw_coefs[1] + (X_test_sis %*% raw_coefs[-1])
p_weighted <- as.vector(1 / (1 + exp(-linear_predictor)))
prob_real <- p_weighted / (p_weighted + 32 * (1 - p_weighted))

evaluate_profit <- function(prob, y, r, lgd, exposure = 1000000) {
  lend <- prob < r / (r + lgd)
  data.frame(
    scenario = sprintf("r=%s%%, LGD=%s%%", r * 100, lgd * 100),
    optimal_threshold = round(r / (r + lgd), 4),
    approved = sum(lend),
    defaults = sum(lend & y == 1),
    net_profit_yuan = sum(lend & y == 0) * exposure * r - sum(lend & y == 1) * exposure * lgd
  )
}
econ <- rbind(evaluate_profit(prob_real, Y_test, 0.05, 1.00),
              evaluate_profit(prob_real, Y_test, 0.06, 0.40))
econ$net_profit_10k <- round(econ$net_profit_yuan / 10000, 2)
write_utf8(econ, "economic_scenarios.csv", row.names = FALSE)
cat("\n[经济效益场景]\n"); print(econ, row.names = FALSE)

# 测试集预测概率（供核对校准与阈值分析）
prob_tab <- data.frame(Y_test = Y_test, prob_weighted = round(p_weighted, 6),
                       prob_calibrated = round(prob_real, 6))
write_utf8(prob_tab, "test_probabilities.csv", row.names = FALSE)

cat("\n导出完成，文件位于", out_dir, "\n")

# ============================================================
# 描述性事实：单变量相关性强度、经济模块内外相关结构、降级展开的变量
# ============================================================
X94 <- as.matrix(train_data$X)
colnames(X94) <- colnames(train_data$X)

# 各特征与目标变量的点二列相关系数
pb_corr <- apply(X94, 2, function(x) cor(x, Y_train))
abs_corr <- abs(pb_corr)
desc <- data.frame(
  item = c("特征数", "|r| 中位数", "|r| 均值", "|r| 最大值", "|r| 最大者",
           "|r|>=0.3 的特征数", "|r|>=0.2 的特征数", "|r|<0.1 的特征数"),
  value = c(ncol(X94), round(median(abs_corr), 4), round(mean(abs_corr), 4),
            round(max(abs_corr), 4), names(which.max(abs_corr)),
            sum(abs_corr >= 0.3), sum(abs_corr >= 0.2), sum(abs_corr < 0.1))
)
write_utf8(desc, "univariate_signal.csv", row.names = FALSE)
cat("\n[单变量信号强度]\n"); print(desc, row.names = FALSE)

top_corr <- data.frame(feature = names(sort(abs_corr, decreasing = TRUE))[1:10],
                       abs_point_biserial_r = round(sort(abs_corr, decreasing = TRUE)[1:10], 4))
write_utf8(top_corr, "top10_univariate_features.csv", row.names = FALSE)
cat("\n[与破产相关性最强的 10 个特征]\n"); print(top_corr, row.names = FALSE)

# 经济模块内外平均绝对相关
# 模块成员直接取自已落盘超高维矩阵的单变量样条列名（形如 Prof.ROA(C)..._ns1），
# 这是建模时真实使用的分块，比脚本里的索引列表更可靠
spline_only <- ultra_names[!is_interaction & is_spline]
spline_block <- sub("^([A-Za-z]+)\\..*$", "\\1", spline_only)
spline_var <- sub("^[A-Za-z]+\\.", "", sub("_(ns[0-9]+|linear)$", "", spline_only))
stopifnot(all(spline_block %in% c("Prof", "Lev", "Cash", "Eff", "Grow")))
# 同一变量有 ns1/ns2/ns3 三列（退化变量仅一列），去重后才是模块内的变量个数
block_members <- lapply(split(spline_var, spline_block), unique)
bn <- c("Prof", "Lev", "Cash", "Eff", "Grow")
block_members <- block_members[bn]
block_size <- data.frame(block = bn,
                         n_features = lengths(block_members),
                         n_spline_cols = as.integer(table(spline_block)[bn]),
                         example = vapply(block_members, function(v) v[1], character(1)))
write_utf8(block_size, "block_membership.csv", row.names = FALSE)
cat("\n[模块规模（自列名解析）]\n"); print(block_size, row.names = FALSE)
cat("模块变量数合计:", sum(block_size$n_features), "\n")

# 与 notebooks/02 中索引列表定义的模块对照，检查两套分块是否一致
nb_idx <- list(
  Prof = c(1:12, 15, 42, 43, 68, 69, 86, 89, 90),
  Lev  = c(14, 35:41, 60, 64, 66, 67, 77:79, 84, 85, 91:93, 94),
  Cash = c(13, 20, 32:34, 54:59, 61:63, 65, 74, 75, 80:83),
  Eff  = c(44:53, 70:73, 76, 87, 88),
  Grow = c(16:19, 21:31)
)
check <- do.call(rbind, lapply(bn, function(b) {
  actual <- block_members[[b]]
  listed <- colnames(X94)[nb_idx[[b]]]
  data.frame(block = b, n_from_names = length(actual), n_from_index = length(listed),
             n_common = length(intersect(actual, listed)),
             only_from_names = length(setdiff(actual, listed)),
             only_from_index = length(setdiff(listed, actual)))
}))
write_utf8(check, "block_membership_check.csv", row.names = FALSE)
cat("\n[模块划分一致性：列名解析 vs notebooks/02 索引]\n"); print(check, row.names = FALSE)

cor_X <- cor(X94)
block_mat <- matrix(NA, length(bn), length(bn), dimnames = list(bn, bn))
for (i in seq_along(bn)) {
  for (j in seq_along(bn)) {
    ii <- match(block_members[[bn[i]]], colnames(X94))
    jj <- match(block_members[[bn[j]]], colnames(X94))
    sub_cor <- abs(cor_X[ii, jj, drop = FALSE])
    block_mat[i, j] <- if (i == j) mean(sub_cor[upper.tri(sub_cor)], na.rm = TRUE) else mean(sub_cor, na.rm = TRUE)
  }
}
write_utf8(round(block_mat, 4), "block_correlation_matrix.csv", row.names = TRUE)
cat("\n[模块级平均绝对相关] 行/列顺序: Prof, Lev, Cash, Eff, Grow\n"); print(round(block_mat, 4))
cat(sprintf("组内平均 %.4f | 组间平均 %.4f\n", mean(diag(block_mat)),
            mean(block_mat[upper.tri(block_mat)])))

# 因节点重合退化为线性项的变量（单变量样条列名以 _linear 结尾）
degraded <- spline_only[grepl("_linear$", spline_only)]
deg_var <- sub("^[A-Za-z]+\\.", "", sub("_linear$", "", degraded))
deg <- data.frame(
  item = c("退化展开的变量", "所属模块", "在 94 维矩阵中的位置", "样条基列数(其余变量)", "样条基列数(合计)"),
  value = c(deg_var, sub("^([A-Za-z]+)\\..*$", "\\1", degraded),
            match(deg_var, colnames(X94)), "3", sum(is_spline))
)
write_utf8(deg, "spline_expansion_note.csv", row.names = FALSE)
cat("\n[样条展开说明]\n"); print(deg, row.names = FALSE)