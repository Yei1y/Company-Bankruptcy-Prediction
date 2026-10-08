# 脚本功能：基于拆分样本的 SCAD 后选择推断
# 与 03_SCAD.R 的区别：将训练集拆分为两半
#   子集 A → SCAD 变量选择（带权重）
#   子集 B → 无权重 logistic 回归推断（p 值、置信区间有效）
# 预测模型仍用全量训练集训练的 SCAD

library(ggplot2)
library(dplyr)
library(ncvreg)
library(pROC)
library(caret)
library(PRROC)
library(ROCR)
library(patchwork)
library(brglm2)

theme_colors <- c("#c6c077", "#8aae76", "#59977a", "#377d79", "#2d636e", "#2f4858")

# ============================================================
# 1. 加载数据
# ============================================================
sis_results <- readRDS("../data/03_SIS_Results_Full.rds")
train_data  <- readRDS("../data/02_Train_Ultrahigh.rds")
test_data   <- readRDS("../data/02_Test_Ultrahigh.rds")

idx <- which(sis_results$selection_freq > 0.7)

# 特征筛选频率分布图
freq_df <- data.frame(
  Feature = idx,
  Frequency = sis_results$selection_freq[idx]
)

p_freq_bar <- ggplot(freq_df, aes(x = Frequency)) +
  geom_histogram(breaks = seq(0.7, 1, by = 0.05),
                 fill = theme_colors[4], color = "white",
                 linewidth = 0.5, alpha = 0.85) +
  scale_x_continuous(breaks = seq(0.7, 1, by = 0.05), limits = c(0.68, 1.02)) +
  scale_y_continuous(labels = scales::label_number(accuracy = 1)) +
  theme_minimal(base_size = 14) +
  labs(title = "Distribution of Highly Stable Features",
       subtitle = "Selection frequency > 0.7 across screening iterations",
       x = "Selection Frequency (Interval = 0.05)",
       y = "Count (Number of Features)") +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 15),
        axis.text.x = element_text(angle = 45, hjust = 1))
print(p_freq_bar)

# ============================================================
# 2. 准备数据
# ============================================================
X_all   <- as.matrix(train_data$X)[, idx, drop = FALSE]
Y_all   <- as.numeric(as.character(train_data$Y))
X_test  <- as.matrix(test_data$X)[, idx, drop = FALSE]
Y_test  <- as.numeric(as.character(test_data$Y))

# 类别权重（用于 SCAD 训练）
n_0 <- sum(Y_all == 0)
n_1 <- sum(Y_all == 1)
weight_ratio <- n_0 / n_1
w_all <- ifelse(Y_all == 1, weight_ratio, 1)

# ============================================================
# 3. 拆分训练集：子集 A（选择）+ 子集 B（推断）
# ============================================================
cat("\n========================================\n")
cat("Split-Sample Post-Selection Inference\n")
cat("========================================\n\n")

set.seed(2024)
n_train <- length(Y_all)
half    <- floor(n_train / 2)
split_idx <- sample(1:n_train, half)

X_A <- X_all[split_idx, , drop = FALSE]
Y_A <- Y_all[split_idx]
w_A <- w_all[split_idx]

X_B <- X_all[-split_idx, , drop = FALSE]
Y_B <- Y_all[-split_idx]

cat(sprintf("Subset A (SCAD selection): n = %d (破产 %d, 健康 %d)\n",
            length(Y_A), sum(Y_A == 1), sum(Y_A == 0)))
cat(sprintf("Subset B (Inference):      n = %d (破产 %d, 健康 %d)\n",
            length(Y_B), sum(Y_B == 1), sum(Y_B == 0)))

# ============================================================
# 4. 子集 A：SCAD 变量选择
# ============================================================
cat("\n[Step 1] SCAD variable selection on Subset A...\n")

set.seed(1)
folds_A  <- createFolds(Y_A, k = 10, list = FALSE)

cv_scad_A <- cv.ncvreg(
  X = X_A, y = Y_A,
  family    = "binomial",
  penalty   = "SCAD",
  gamma     = 3.7,
  nlambda   = 150,
  eps       = 0.01,
  fold      = folds_A,
  max.iter  = 10000,
  obs.weights = w_A,
  returnY   = TRUE
)

coef_A       <- coef(cv_scad_A)
selected_vars <- which(coef_A[-1] != 0)

cat(sprintf("  Optimal lambda: %.6f\n", cv_scad_A$lambda.min))
cat(sprintf("  SCAD selected %d features from Subset A.\n", length(selected_vars)))

if (length(selected_vars) == 0) {
  cat("ERROR: SCAD selected no variables. Cannot proceed with inference.\n")
  stop("No variables selected.")
}

# ============================================================
# 5. 子集 B：无权重 logistic 回归（valid 推断）
# ============================================================
cat("\n[Step 2] Unweighted logistic regression on Subset B (inference)...\n")

X_B_sel <- X_B[, selected_vars, drop = FALSE]
df_B    <- data.frame(y = Y_B, X_B_sel)

# 短变量名映射
var_mapping <- data.frame(
  ShortID  = paste0("F", sprintf("%02d", seq_along(selected_vars))),
  Original = colnames(X_B_sel),
  stringsAsFactors = FALSE
)

# 第一轮：全变量 glm，检测分离
fit_B_all <- glm(y ~ ., data = df_B, family = binomial)
s_all     <- summary(fit_B_all)$coefficients
se_all    <- s_all[, "Std. Error"]

var_in_model <- rownames(s_all)[rownames(s_all) != "(Intercept)"]
aliased_vars <- setdiff(colnames(X_B_sel), var_in_model)
se_threshold <- 10
unstable_vars <- var_in_model[
  sapply(var_in_model, function(vn) {
    i <- which(rownames(s_all) == vn)
    length(i) > 0 && (se_all[i] > se_threshold || is.na(se_all[i]))
  })
]
vars_to_remove <- c(aliased_vars, unstable_vars)

if (length(vars_to_remove) > 0) {
  keep_mask <- !(var_mapping$Original %in% vars_to_remove)
  cat(sprintf("  Removed %d variable(s) (aliased or Std_Error > %d)\n",
              length(vars_to_remove), se_threshold))

  selected_vars <- selected_vars[keep_mask]
  X_B_sel       <- X_B_sel[, keep_mask, drop = FALSE]
  var_mapping   <- var_mapping[keep_mask, , drop = FALSE]
  df_B          <- data.frame(y = Y_B, X_B_sel)
  fit_B         <- glm(y ~ ., data = df_B, family = binomial)
} else {
  fit_B <- fit_B_all
}

# 提取推断结果
s_B    <- summary(fit_B)$coefficients
est_B  <- s_B[, "Estimate"]
se_B   <- s_B[, "Std. Error"]
z_B    <- s_B[, "z value"]
p_B    <- s_B[, "Pr(>|z|)"]

# 用短 ID 替换变量名
var_names_B <- rownames(s_B)
for (i in seq_along(var_names_B)) {
  if (var_names_B[i] != "(Intercept)") {
    m <- which(var_mapping$Original == var_names_B[i])
    if (length(m) > 0) var_names_B[i] <- var_mapping$ShortID[m]
  }
}

alpha_ci <- 0.05
z_crit   <- qnorm(1 - alpha_ci / 2)
ci_lower_B <- est_B - z_crit * se_B
ci_upper_B <- est_B + z_crit * se_B

# ============================================================
# 6. Bootstrap（在子集 B 上，无权重）
# ============================================================
cat("\n[Step 3] Bootstrap validation on Subset B (B = 200)...\n")

n_boot     <- 200
boot_coefs <- matrix(NA, n_boot, length(est_B))
colnames(boot_coefs) <- var_names_B

for (b in seq_len(n_boot)) {
  boot_idx  <- sample(nrow(X_B_sel), replace = TRUE)
  df_boot   <- data.frame(y = Y_B[boot_idx], X_B_sel[boot_idx, , drop = FALSE])

  tryCatch({
    boot_fit <- glm(y ~ ., data = df_boot, family = binomial)
    boot_coefs[b, ] <- coef(boot_fit)
  }, error = function(e) {
    tryCatch({
      boot_fit <- glm(y ~ ., data = df_boot, family = binomial,
                      method = "brglmFit")
      boot_coefs[b, ] <- coef(boot_fit)
    }, error = function(e2) {
      boot_coefs[b, ] <<- NA
    })
  })

  if (b %% 50 == 0) cat(sprintf("  Bootstrap %d / %d\n", b, n_boot))
}

boot_ci_lower <- boot_ci_upper <- rep(NA, length(est_B))
names(boot_ci_lower) <- names(boot_ci_upper) <- var_names_B

for (i in seq_along(est_B)) {
  s_i <- boot_coefs[, i]
  s_i <- s_i[!is.na(s_i)]
  if (length(s_i) > 10) {
    boot_ci_lower[i] <- quantile(s_i, alpha_ci / 2)
    boot_ci_upper[i] <- quantile(s_i, 1 - alpha_ci / 2)
  }
}

# ============================================================
# 7. 组装推断结果表
# ============================================================
inference_results <- data.frame(
  Variable           = var_names_B,
  Coefficient        = round(est_B, 5),
  Std_Error          = round(se_B, 5),
  Z_Statistic        = round(z_B, 4),
  P_Value            = round(p_B, 5),
  Significant        = p_B < alpha_ci,
  Odds_Ratio         = round(exp(est_B), 4),
  OR_CI_Lower        = round(exp(ci_lower_B), 4),
  OR_CI_Upper        = round(exp(ci_upper_B), 4),
  Analytic_CI_Lower  = round(ci_lower_B, 5),
  Analytic_CI_Upper  = round(ci_upper_B, 5),
  Bootstrap_CI_Lower = round(boot_ci_lower, 5),
  Bootstrap_CI_Upper = round(boot_ci_upper, 5),
  stringsAsFactors   = FALSE
)

cat("\n====== Split-Sample Inference Results ======\n")
print(inference_results, row.names = FALSE)

cat("\n--- Variable ID Mapping ---\n")
print(var_mapping, row.names = FALSE)

# 显著性汇总
has_intercept <- any(inference_results$Variable == "(Intercept)")
n_features    <- nrow(inference_results) - as.integer(has_intercept)
sig_flag      <- inference_results$Significant
cat(sprintf("\nSignificant at alpha = 0.05: %d / %d variables\n",
            sum(sig_flag[inference_results$Variable != "(Intercept)"]), n_features))

# 推断模型诊断
cat(sprintf("\nRefit model diagnostics:\n"))
cat(sprintf("  Residual deviance: %.2f on %d df\n", deviance(fit_B), df.residual(fit_B)))
cat(sprintf("  AIC: %.2f\n", AIC(fit_B)))

# ============================================================
# 8. 置信区间可视化
# ============================================================
plot_data <- inference_results
if (has_intercept) plot_data <- plot_data[plot_data$Variable != "(Intercept)", ]

if (nrow(plot_data) > 0) {
  plot_data$Variable <- reorder(plot_data$Variable, plot_data$Coefficient)

  # 系数 + Wald CI
  p_analytic <- ggplot(plot_data, aes(x = Variable, y = Coefficient)) +
    geom_point(aes(color = Significant), size = 3) +
    geom_errorbar(aes(ymin = Analytic_CI_Lower, ymax = Analytic_CI_Upper,
                      color = Significant), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    scale_color_manual(values = c("TRUE" = theme_colors[4], "FALSE" = theme_colors[2]),
                       labels = c("Significant", "Not Significant")) +
    coord_flip() + theme_minimal(base_size = 14) +
    labs(title = "Post-SCAD Coefficients (Split-Sample Inference)",
         subtitle = "95% Wald CI | Subset B (no selection bias)",
         x = "Variable", y = "Coefficient (Log-Odds)") +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 15),
          legend.position = "bottom")
  print(p_analytic)

  # 优势比森林图
  p_or <- ggplot(plot_data, aes(x = Variable, y = Odds_Ratio)) +
    geom_point(aes(color = Significant), size = 3) +
    geom_errorbar(aes(ymin = OR_CI_Lower, ymax = OR_CI_Upper,
                      color = Significant), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    scale_color_manual(values = c("TRUE" = theme_colors[4], "FALSE" = theme_colors[2]),
                       labels = c("Significant", "Not Significant")) +
    coord_flip() + theme_minimal(base_size = 14) +
    labs(title = "Odds Ratios with 95% Confidence Intervals",
         subtitle = "Split-Sample: no post-selection bias",
         x = "Variable", y = "Odds Ratio") +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 15),
          legend.position = "bottom")
  print(p_or)

  # Bootstrap CI
  if (any(!is.na(plot_data$Bootstrap_CI_Lower))) {
    p_boot <- ggplot(plot_data, aes(x = Variable, y = Coefficient)) +
      geom_point(aes(color = Significant), size = 3) +
      geom_errorbar(aes(ymin = Bootstrap_CI_Lower, ymax = Bootstrap_CI_Upper,
                        color = Significant), width = 0.2, alpha = 0.7) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      scale_color_manual(values = c("TRUE" = theme_colors[5], "FALSE" = theme_colors[3]),
                         labels = c("Significant", "Not Significant")) +
      coord_flip() + theme_minimal(base_size = 14) +
      labs(title = "Bootstrap-Validated Confidence Intervals",
           subtitle = "95% Bootstrap Percentile CI (B = 200) on Subset B",
           x = "Variable", y = "Coefficient (Log-Odds)") +
      theme(panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", size = 15),
            legend.position = "bottom")
    print(p_boot)
  }
}

# ============================================================
# 9. 全量 SCAD 预测模型（用于测试集评估）
# ============================================================
cat("\n[Step 4] Full-data SCAD model for prediction...\n")

set.seed(1)
folds_full <- createFolds(Y_all, k = 10, list = FALSE)

cv_scad_full <- cv.ncvreg(
  X = X_all, y = Y_all,
  family    = "binomial",
  penalty   = "SCAD",
  gamma     = 3.7,
  nlambda   = 150,
  eps       = 0.01,
  fold      = folds_full,
  max.iter  = 10000,
  obs.weights = w_all,
  returnY   = TRUE
)

coef_full <- coef(cv_scad_full)
active_full <- sum(coef_full[-1] != 0)
cat(sprintf("  Full SCAD: lambda.min = %.6f, active features = %d\n",
            cv_scad_full$lambda.min, active_full))

# ============================================================
# 10. 测试集评估
# ============================================================
cat("\n[Step 5] Test set evaluation...\n")

prob_pred <- predict(cv_scad_full, X_test, type = "response")

# ROC
roc_obj    <- roc(response = Y_test, predictor = as.numeric(prob_pred),
                  levels = c(0, 1), direction = "<", quiet = TRUE)
auc_value  <- auc(roc_obj)
best_info  <- coords(roc_obj, "best",
                     ret = c("threshold", "specificity", "sensitivity", "accuracy"))
optimal_threshold <- best_info$threshold[1]

# PR-AUC
fg <- prob_pred[Y_test == 1]
bg <- prob_pred[Y_test == 0]
pr_obj       <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
pr_auc_value <- pr_obj$auc.integral

# KS
pred_rocr <- prediction(prob_pred, Y_test)
perf_tpr  <- performance(pred_rocr, "tpr")
perf_fpr  <- performance(pred_rocr, "fpr")
cutoffs   <- perf_tpr@x.values[[1]]
tpr_vals  <- perf_tpr@y.values[[1]]
fpr_vals  <- perf_fpr@y.values[[1]]
ks_diff   <- tpr_vals - fpr_vals
ks_stat   <- max(ks_diff)
ks_index  <- which.max(ks_diff)

# 混淆矩阵
class_pred  <- factor(ifelse(prob_pred >= optimal_threshold, 1, 0), levels = c(0, 1))
Y_test_f    <- factor(Y_test, levels = c(0, 1))
conf_matrix <- confusionMatrix(class_pred, Y_test_f, positive = "1")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall    <- conf_matrix$byClass["Sensitivity"]
f1_score  <- conf_matrix$byClass["F1"]
if (is.na(precision)) precision <- 0
if (is.na(recall))    recall    <- 0
f2_score <- (1 + 2^2) * (precision * recall) / ((2^2 * precision) + recall)
if (is.na(f2_score)) f2_score <- 0

# ============================================================
# 11. 评估可视化
# ============================================================
roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = theme_colors[4], linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "ROC Curve", subtitle = sprintf("AUC = %.4f", auc_value),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

pr_df <- data.frame(Recall = pr_obj$curve[, 1], Precision = pr_obj$curve[, 2])
p_pr <- ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(color = theme_colors[5], linewidth = 1.2) +
  theme_minimal(base_size = 14) +
  labs(title = "Precision-Recall Curve",
       subtitle = sprintf("PR-AUC = %.4f", pr_auc_value),
       x = "Recall (Sensitivity)", y = "Precision (PPV)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

ks_df <- data.frame(Threshold = cutoffs, TPR = tpr_vals, FPR = fpr_vals)
p_ks <- ggplot(ks_df, aes(x = Threshold)) +
  geom_line(aes(y = TPR, color = "TPR"), linewidth = 1.2) +
  geom_line(aes(y = FPR, color = "FPR"), linewidth = 1.2) +
  geom_vline(xintercept = cutoffs[ks_index], linetype = "dashed", color = "black") +
  scale_color_manual(name = "Metrics",
                     values = c("TPR" = theme_colors[4], "FPR" = theme_colors[2])) +
  theme_minimal(base_size = 14) +
  labs(title = "KS Curve", subtitle = sprintf("KS Statistic = %.4f", ks_stat),
       x = "Probability Threshold", y = "Cumulative Rate") +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 15),
        legend.position = "bottom")

calib_data <- data.frame(
  obs = factor(ifelse(Y_test == 1, "Bankruptcy", "Healthy"),
               levels = c("Bankruptcy", "Healthy")),
  Bankruptcy = prob_pred
)
calib_obj <- calibration(obs ~ Bankruptcy, data = calib_data,
                         class = "Bankruptcy", cuts = 10)
calib_df  <- calib_obj$data
p_calib <- ggplot(calib_df, aes(x = midpoint, y = Percent)) +
  geom_line(color = theme_colors[6], linewidth = 1.2) +
  geom_point(color = theme_colors[6], size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "Calibration Curve (Reliability Diagram)",
       x = "Bin Midpoint (Predicted Probability %)",
       y = "Observed Event Percentage (%)") +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 15))

# SCAD 系数图（全量模型）
all_coefs    <- coef(cv_scad_full)
active_coefs <- all_coefs[-1][all_coefs[-1] != 0]

if (length(active_coefs) > 0) {
  coef_df <- data.frame(Feature_Name = names(active_coefs), Coefficient = active_coefs)
  coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
  coef_df$Feature_ID <- paste0("F", sprintf("%02d", 1:nrow(coef_df)))
  coef_df$Feature_ID <- factor(coef_df$Feature_ID, levels = coef_df$Feature_ID)

  p_coef <- ggplot(coef_df, aes(x = Feature_ID, y = Coefficient, fill = Coefficient > 0)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_fill_manual(values = c("TRUE" = theme_colors[4], "FALSE" = theme_colors[2]),
                      guide = "none") +
    coord_flip() + theme_minimal(base_size = 14) +
    labs(title = "SCAD Feature Importance (Full Model)",
         subtitle = "See mapping table for feature definitions",
         x = "Feature ID", y = "Coefficient Value") +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 15))
  print(p_coef)

  cat("\n--- Full Model Feature Mapping ---\n")
  print(coef_df[, c("Feature_ID", "Feature_Name", "Coefficient")], row.names = FALSE)
}

# 组合评估图
p_combined <- (p_roc | p_pr) /
  (p_ks | p_calib) +
  plot_annotation(
    title = "Comprehensive Model Evaluation Metrics",
    theme = theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5))
  )
print(p_combined)

# ============================================================
# 12. 评估指标汇总
# ============================================================
eval_metrics_df <- data.frame(
  Metric = c("ROC-AUC", "PR-AUC", "KS Statistic", "Optimal Threshold",
             "Precision (PPV)", "Recall (Sensitivity)", "F1-Score", "F2-Score"),
  Value = round(c(auc_value, pr_auc_value, ks_stat, optimal_threshold,
                  precision, recall, f1_score, f2_score), 4)
)

cat("\n--- Model Evaluation Metrics ---\n")
print(eval_metrics_df, row.names = FALSE)
cat("\n--- Confusion Matrix (Optimal Threshold) ---\n")
print(conf_matrix$table)

# ============================================================
# 13. 推断对比：Split-sample vs 全量 refit
# ============================================================
cat("\n========================================\n")
cat("Comparison: Split-sample vs Full-data refit\n")
cat("========================================\n")

# 在全量数据上做一次无权重 refit 用于对比
X_full_sel <- X_all[, selected_vars, drop = FALSE]
df_full    <- data.frame(y = Y_all, X_full_sel)
fit_full_refit <- glm(y ~ ., data = df_full, family = binomial)
s_full <- summary(fit_full_refit)$coefficients

# 用短 ID 替换
var_names_full <- rownames(s_full)
for (i in seq_along(var_names_full)) {
  if (var_names_full[i] != "(Intercept)") {
    m <- which(var_mapping$Original == var_names_full[i])
    if (length(m) > 0) var_names_full[i] <- var_mapping$ShortID[m]
  }
}

compare_df <- data.frame(
  Variable      = var_names_B,
  Split_PValue  = round(p_B, 5),
  Full_PValue   = round(s_full[, "Pr(>|z|)"], 5),
  Split_Coef    = round(est_B, 4),
  Full_Coef     = round(s_full[, "Estimate"], 4)
)
cat("\nCoefficient & p-value comparison:\n")
print(compare_df, row.names = FALSE)

cat("\nInterpretation: If Split_PValue > Full_PValue for most variables,\n")
cat("the full-data refit was over-optimistic (inflated by selection bias).\n")

# ============================================================
# 14. 保存结果
# ============================================================
saveRDS(inference_results, "../data/03_SCAD_Split_Inference.rds")
saveRDS(cv_scad_full,      "../data/03_SCAD_Model.rds")

save(prob_pred, roc_obj, pr_obj, conf_matrix, eval_metrics_df, Y_test,
     inference_results, cv_scad_full, cv_scad_A, selected_vars, var_mapping,
     file = "../data/03_Split_Evaluation_Metrics.RData")

cat("\nAll results saved. Script completed.\n")
