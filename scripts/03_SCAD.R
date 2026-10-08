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

# 加载数据并提取 SIS 筛选后的特征
sis_results <- readRDS("../data/03_SIS_Results_Full.rds")
train_data  <- readRDS("../data/02_Train_Ultrahigh.rds")
test_data   <- readRDS("../data/02_Test_Ultrahigh.rds")

idx <- which(sis_results$selection_freq > 0.7)

# 绘制特征选择频次分布
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

# 构建数据集并训练 SCAD 模型
X_train_sis <- as.matrix(train_data$X)[, idx, drop = FALSE]
X_test_sis  <- as.matrix(test_data$X)[, idx, drop = FALSE]
Y_train_bin <- as.numeric(as.character(train_data$Y))
Y_test_bin  <- as.numeric(as.character(test_data$Y))

set.seed(1)

n_0 <- sum(Y_train_bin == 0)
n_1 <- sum(Y_train_bin == 1)
weight_ratio <- n_0 / n_1
w <- ifelse(Y_train_bin == 1, weight_ratio, 1)

folds_index <- createFolds(Y_train_bin, k = 10, list = FALSE)

cv_scad <- cv.ncvreg(
  X = X_train_sis,
  y = Y_train_bin,
  family = "binomial",
  penalty = "SCAD",
  gamma = 3.7,
  nlambda = 150,
  eps = 0.01,
  fold = folds_index,
  max.iter = 10000,
  obs.weights = w,
  returnY = TRUE
)

best_lambda <- cv_scad$lambda.min
coef_scad <- coef(cv_scad)
active_vars <- sum(coef_scad[-1] != 0)

cat("\n[SCAD Model Summary]\n")
cat("Optimal lambda :", best_lambda, "\n")
cat("Active features:", active_vars, "\n")

saveRDS(cv_scad, "../data/03_SCAD_Model.rds")

# 准备模型评估数据 (ROC, PR, KS, F 分数)
prob_pred <- predict(cv_scad, X_test_sis, type = "response")

# 计算 ROC 和 Youden 阈值
roc_obj <- roc(response = Y_test_bin, predictor = as.numeric(prob_pred),
               levels = c(0, 1), direction = "<", quiet = TRUE)
auc_value <- auc(roc_obj)
best_thresh_info <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity", "accuracy"))
optimal_threshold <- best_thresh_info$threshold[1]

# 计算 PR 曲线
fg <- prob_pred[Y_test_bin == 1]
bg <- prob_pred[Y_test_bin == 0]
pr_obj <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
pr_auc_value <- pr_obj$auc.integral

# 计算 KS 统计量
pred_rocr <- prediction(prob_pred, Y_test_bin)
perf_tpr <- performance(pred_rocr, "tpr")
perf_fpr <- performance(pred_rocr, "fpr")
cutoffs <- perf_tpr@x.values[[1]]
tpr_vals <- perf_tpr@y.values[[1]]
fpr_vals <- perf_fpr@y.values[[1]]
ks_diff <- tpr_vals - fpr_vals
ks_stat <- max(ks_diff)
ks_index <- which.max(ks_diff)

# 计算混淆矩阵与 F 分数
class_pred <- factor(ifelse(prob_pred >= optimal_threshold, 1, 0), levels = c(0, 1))
Y_test_factor <- factor(Y_test_bin, levels = c(0, 1))
conf_matrix <- confusionMatrix(class_pred, Y_test_factor, positive = "1")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- conf_matrix$byClass["F1"]

if(is.na(precision)) precision <- 0
if(is.na(recall)) recall <- 0

f2_score <- (1 + 2^2) * (precision * recall) / ((2^2 * precision) + recall)
if(is.na(f2_score)) f2_score <- 0

# 绘制 ROC 曲线
roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = theme_colors[4], linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "ROC Curve", subtitle = sprintf("AUC = %.4f", auc_value),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 绘制 Precision-Recall 曲线
pr_df <- data.frame(Recall = pr_obj$curve[, 1], Precision = pr_obj$curve[, 2])
p_pr <- ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(color = theme_colors[5], linewidth = 1.2) +
  theme_minimal(base_size = 14) +
  labs(title = "Precision-Recall Curve", subtitle = sprintf("PR-AUC = %.4f", pr_auc_value),
       x = "Recall (Sensitivity)", y = "Precision (PPV)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 绘制 KS 曲线
ks_df <- data.frame(Threshold = cutoffs, TPR = tpr_vals, FPR = fpr_vals)
p_ks <- ggplot(ks_df, aes(x = Threshold)) +
  geom_line(aes(y = TPR, color = "TPR"), linewidth = 1.2) +
  geom_line(aes(y = FPR, color = "FPR"), linewidth = 1.2) +
  geom_vline(xintercept = cutoffs[ks_index], linetype = "dashed", color = "black") +
  scale_color_manual(name = "Metrics", values = c("TPR" = theme_colors[4], "FPR" = theme_colors[2])) +
  theme_minimal(base_size = 14) +
  labs(title = "KS Curve", subtitle = sprintf("KS Statistic = %.4f", ks_stat),
       x = "Probability Threshold", y = "Cumulative Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15), legend.position = "bottom")

# 绘制校准曲线
calib_data <- data.frame(
  obs = factor(ifelse(Y_test_bin == 1, "Bankruptcy", "Healthy"), levels = c("Bankruptcy", "Healthy")),
  Bankruptcy = prob_pred
)
calib_obj <- calibration(obs ~ Bankruptcy, data = calib_data, class = "Bankruptcy", cuts = 10)
calib_df <- calib_obj$data

p_calib <- ggplot(calib_df, aes(x = midpoint, y = Percent)) +
  geom_line(color = theme_colors[6], linewidth = 1.2) +
  geom_point(color = theme_colors[6], size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "Calibration Curve (Reliability Diagram)",
       x = "Bin Midpoint (Predicted Probability %)", y = "Observed Event Percentage (%)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 绘制特征系数重要性图（编号映射避免长变量名）
all_coefs <- coef(cv_scad)
active_coefs <- all_coefs[-1][all_coefs[-1] != 0]

if(length(active_coefs) > 0) {
  coef_df <- data.frame(Feature_Name = names(active_coefs), Coefficient = active_coefs)
  coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]

  # 生成两位数编号 (F01, F02...)
  num_active <- nrow(coef_df)
  coef_df$Feature_ID <- paste0("F", sprintf("%02d", 1:num_active))

  coef_df$Feature_ID <- factor(coef_df$Feature_ID, levels = coef_df$Feature_ID)

  # 提取编号-原始名映射表供报告使用
  feature_mapping_table <- coef_df[, c("Feature_ID", "Feature_Name", "Coefficient")]

  p_coef <- ggplot(coef_df, aes(x = Feature_ID, y = Coefficient, fill = Coefficient > 0)) +
    geom_bar(stat = "identity", width = 0.6) +
    scale_fill_manual(values = c("TRUE" = theme_colors[4], "FALSE" = theme_colors[2]), guide = "none") +
    coord_flip() +
    theme_minimal(base_size = 14) +
    labs(title = "SCAD Feature Importance (Mapped)",
         subtitle = "See table below for feature definitions",
         x = "Feature ID", y = "Coefficient Value") +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 15))

  print(p_coef)

  cat("\n--- SCAD Feature ID Mapping Table ---\n")
  print(feature_mapping_table, row.names = FALSE)
  cat("--------------------------------------\n")
}

# 组合并输出所有评估图表
p_combined <- (p_roc | p_pr) /
  (p_ks | p_calib) +
  plot_annotation(
    title = "Comprehensive Model Evaluation Metrics",
    theme = theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5))
  )
print(p_combined)
if(exists("p_coef")) print(p_coef)

# 构建模型评估指标汇总表
eval_metrics_df <- data.frame(
  Metric = c("ROC-AUC", "PR-AUC", "KS Statistic", "Optimal Threshold",
             "Precision (PPV)", "Recall (Sensitivity)", "F1-Score", "F2-Score"),
  Value = round(c(auc_value, pr_auc_value, ks_stat, optimal_threshold,
                  precision, recall, f1_score, f2_score), 4)
)

# 后选择推断：SCAD 筛选后拟合标准 Logistic 回归并计算置信区间
# SCAD 完成高维变量筛选后，对选中的变量拟合标准 Logistic 回归，
# 获得系数估计、标准误、p 值、置信区间。
# 检测"分离"变量 (Std_Error > 10) 并标记其不可靠性，不强行使用 brglm2 导致全局退化。

# 核心推断函数：SCAD 筛选 → 标准 Logistic 回归 → Bootstrap 验证
post_scad_inference <- function(cv_scad_obj, X_train, y_train, weights = NULL,
                                alpha = 0.05, bootstrap = TRUE, n_boot = 200,
                                se_threshold = 10) {

  coef_all <- coef(cv_scad_obj)
  active_idx <- which(coef_all[-1] != 0)

  # 按 SCAD 系数绝对值降序排列
  coef_nonzero <- coef_all[-1][active_idx]
  active_idx <- active_idx[order(abs(coef_nonzero), decreasing = TRUE)]

  if (length(active_idx) == 0) {
    cat("SCAD selected no variables. Inference skipped.\n")
    return(NULL)
  }

  n_selected_initial <- length(active_idx)
  cat(sprintf("SCAD selected %d variables. Fitting standard logistic...\n",
              n_selected_initial))

  X_selected <- X_train[, active_idx, drop = FALSE]

  # 生成短变量名映射 (F01, F02, ...)
  var_mapping <- data.frame(
    ShortID  = paste0("F", sprintf("%02d", seq_along(active_idx))),
    Original = colnames(X_selected),
    stringsAsFactors = FALSE
  )

  df <- data.frame(y = y_train, X_selected)

  # 第一轮：全变量 glm，检测共线别名化和分离退化
  fit_glm_all <- if (!is.null(weights)) {
    glm(y ~ ., data = df, family = binomial, weights = weights)
  } else {
    glm(y ~ ., data = df, family = binomial)
  }

  s_all <- summary(fit_glm_all)$coefficients
  var_names_all <- rownames(s_all)
  se_all <- s_all[, "Std. Error"]

  # 标记别名化变量（glm 因共线自动丢弃）和标准误过大的变量（分离信号）
  var_in_model_all <- var_names_all[var_names_all != "(Intercept)"]
  aliased_vars <- setdiff(colnames(X_selected), var_in_model_all)
  unstable_vars <- var_in_model_all[
    sapply(var_in_model_all, function(vn) {
      idx <- which(rownames(s_all) == vn)
      length(idx) > 0 && (se_all[idx] > se_threshold || is.na(se_all[idx]))
    })
  ]
  vars_to_remove <- c(aliased_vars, unstable_vars)

  if (length(vars_to_remove) > 0) {
    remove_mask <- var_mapping$Original %in% vars_to_remove
    cat(sprintf("  Removed %d variable(s) (aliased or Std_Error > %d): %s\n",
                length(vars_to_remove), se_threshold,
                paste(var_mapping$ShortID[remove_mask], collapse = ", ")))
    cat("  Refitting standard logistic on reduced variable set...\n")

    keep <- !remove_mask
    active_idx <- active_idx[keep]
    X_selected <- X_selected[, keep, drop = FALSE]
    var_mapping <- var_mapping[keep, , drop = FALSE]

    df <- data.frame(y = y_train, X_selected)
    fit_glm <- if (!is.null(weights)) {
      glm(y ~ ., data = df, family = binomial, weights = weights)
    } else {
      glm(y ~ ., data = df, family = binomial)
    }
  } else {
    fit_glm <- fit_glm_all
  }

  # 从标准 Logistic 回归提取系数推断结果
  s <- summary(fit_glm)$coefficients
  var_names <- rownames(s)
  # 用短 ID 替代原始长变量名
  for (i in seq_along(var_names)) {
    if (var_names[i] != "(Intercept)") {
      matched <- which(var_mapping$Original == var_names[i])
      if (length(matched) > 0) var_names[i] <- var_mapping$ShortID[matched]
    }
  }
  est <- s[, "Estimate"]
  se  <- s[, "Std. Error"]
  z_val <- s[, "z value"]
  p_val <- s[, "Pr(>|z|)"]

  # 第二轮分离诊断（应无异常）
  unstable_flag <- se > se_threshold | is.na(se)
  if (any(unstable_flag)) {
    cat("  Refit still has Std_Error >", se_threshold, ":",
        paste(var_names[unstable_flag], collapse = ", "), "\n")
  }

  z_crit <- qnorm(1 - alpha / 2)
  ci_lower <- est - z_crit * se
  ci_upper <- est + z_crit * se

  # Bootstrap 验证（在无惩罚模型上进行）
  boot_ci_lower <- boot_ci_upper <- rep(NA, length(est))
  names(boot_ci_lower) <- names(boot_ci_upper) <- var_names

  if (bootstrap) {
    cat(sprintf("Bootstrap validation on refitted model (B = %d)...\n", n_boot))
    boot_coefs <- matrix(NA, n_boot, length(est))
    colnames(boot_coefs) <- var_names

    for (b in seq_len(n_boot)) {
      idx <- sample(nrow(X_train), replace = TRUE)
      df_boot <- data.frame(y = y_train[idx], X_selected[idx, , drop = FALSE])

      tryCatch({
        w_boot <- if (!is.null(weights)) weights[idx] else NULL
        boot_fit <- if (!is.null(w_boot)) {
          glm(y ~ ., data = df_boot, family = binomial, weights = w_boot)
        } else {
          glm(y ~ ., data = df_boot, family = binomial)
        }
        boot_coefs[b, ] <- coef(boot_fit)
      }, error = function(e) {
        # 标准 glm 因分离失败时降级为 brglm2
        tryCatch({
          boot_fit <- if (!is.null(w_boot)) {
            glm(y ~ ., data = df_boot, family = binomial,
                weights = w_boot, method = "brglmFit")
          } else {
            glm(y ~ ., data = df_boot, family = binomial,
                method = "brglmFit")
          }
          boot_coefs[b, ] <- coef(boot_fit)
        }, error = function(e2) {
          boot_coefs[b, ] <- NA
        })
      })

      if (b %% 50 == 0) cat(sprintf("  Bootstrap %d / %d\n", b, n_boot))
    }

    for (i in seq_along(est)) {
      s_i <- boot_coefs[, i]
      s_i <- s_i[!is.na(s_i)]
      if (length(s_i) > 0) {
        boot_ci_lower[i] <- quantile(s_i, alpha / 2, na.rm = TRUE)
        boot_ci_upper[i] <- quantile(s_i, 1 - alpha / 2, na.rm = TRUE)
      }
    }
  }

  # 组装结果
  results <- data.frame(
    Variable          = var_names,
    Coefficient       = round(est, 5),
    Std_Error         = round(se, 5),
    Z_Statistic       = round(z_val, 4),
    P_Value           = round(p_val, 5),
    Significant       = p_val < alpha,
    Odds_Ratio        = round(exp(est), 4),
    OR_CI_Lower       = round(exp(ci_lower), 4),
    OR_CI_Upper       = round(exp(ci_upper), 4),
    Analytic_CI_Lower = round(ci_lower, 5),
    Analytic_CI_Upper = round(ci_upper, 5),
    Bootstrap_CI_Lower = round(boot_ci_lower, 5),
    Bootstrap_CI_Upper = round(boot_ci_upper, 5),
    stringsAsFactors  = FALSE
  )

  attr(results, "selected_idx")    <- active_idx
  attr(results, "glm_fit")         <- fit_glm
  attr(results, "var_mapping")     <- var_mapping
  attr(results, "n_scad_selected") <- n_selected_initial
  return(results)
}

# 执行后选择推断
cat("Post-Selection Significance Testing\n")
cat("Pipeline: SIS screening -> SCAD selection -> Standard logistic refit\n")
cat("          (Bootstrap with brglm2 fallback for separation)\n\n")

inference_results <- post_scad_inference(
  cv_scad, X_train_sis, Y_train_bin,
  weights = w, bootstrap = TRUE, n_boot = 200
)

# 推断成功时展示结果并绘图
if (!is.null(inference_results)) {

  cat("\n====== Post-SCAD Inference Results ======\n")
  print(inference_results, row.names = FALSE)

  # 打印短 ID 与原始变量名映射
  var_map <- attr(inference_results, "var_mapping")
  if (!is.null(var_map)) {
    cat("\n--- Variable ID Mapping (short ID -> original name) ---\n")
    print(var_map, row.names = FALSE)
    cat("------------------------------------------------------\n")
  }

  # 显著性汇总（排除截距项）
  has_intercept <- any(inference_results$Variable == "(Intercept)")
  n_features <- nrow(inference_results) - as.integer(has_intercept)
  sig_flag <- inference_results$Significant
  cat(sprintf("\nSignificant at alpha = 0.05 :  %d / %d variables\n",
              sum(sig_flag[inference_results$Variable != "(Intercept)"]), n_features))
  if (sum(sig_flag) > 0) {
    cat("Significant:", paste(inference_results$Variable[sig_flag], collapse = ", "), "\n")
  }

  # 置信区间可视化（排除截距项）
  has_intercept <- any(inference_results$Variable == "(Intercept)")
  plot_data <- inference_results
  if (has_intercept) plot_data <- plot_data[plot_data$Variable != "(Intercept)", ]

  if (nrow(plot_data) > 0) {

    # 系数图 + 解析 CI
    plot_data$Variable <- reorder(plot_data$Variable, plot_data$Coefficient)

    p_analytic <- ggplot(plot_data, aes(x = Variable, y = Coefficient)) +
      geom_point(aes(color = Significant), size = 3) +
      geom_errorbar(aes(ymin = Analytic_CI_Lower, ymax = Analytic_CI_Upper,
                        color = Significant), width = 0.2) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
      scale_color_manual(values = c("TRUE" = theme_colors[4], "FALSE" = theme_colors[2]),
                         labels = c("Significant", "Not Significant")) +
      coord_flip() + theme_minimal(base_size = 14) +
      labs(title = "Post-SCAD Coefficients (Unpenalized Refit)",
           subtitle = "95% Confidence Intervals (Wald)",
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
           subtitle = "OR > 1 -> positive association with outcome",
           x = "Variable", y = "Odds Ratio") +
      theme(panel.grid.minor = element_blank(),
            plot.title = element_text(face = "bold", size = 15),
            legend.position = "bottom")
    print(p_or)

    # Bootstrap CI（若有）
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
             subtitle = "95% Bootstrap Percentile CI (B = 200)",
             x = "Variable", y = "Coefficient (Log-Odds)") +
        theme(panel.grid.minor = element_blank(),
              plot.title = element_text(face = "bold", size = 15),
              legend.position = "bottom")
      print(p_boot)
    }
  }

  # 模型稳定性评估
  cat("\n=== Model Stability (Bootstrap vs Analytic CI) ===\n")
  if (any(!is.na(inference_results$Bootstrap_CI_Lower))) {
    ci_comp <- data.frame(
      Variable       = inference_results$Variable,
      Analytic_Width = inference_results$Analytic_CI_Upper - inference_results$Analytic_CI_Lower,
      Bootstrap_Width = inference_results$Bootstrap_CI_Upper - inference_results$Bootstrap_CI_Lower
    )
    ci_comp$Width_Ratio <- ci_comp$Bootstrap_Width / ci_comp$Analytic_Width
    cat("CI width comparison:\n")
    print(ci_comp, row.names = FALSE)

    avg_ratio <- mean(ci_comp$Width_Ratio, na.rm = TRUE)
    cat(sprintf("Average Bootstrap/Analytic width ratio: %.3f\n", avg_ratio))
    if (avg_ratio > 1.2) {
      cat("-> Bootstrap CIs are wider -> model may be slightly unstable.\n")
    } else if (avg_ratio < 0.8) {
      cat("-> Bootstrap CIs are narrower.\n")
    } else {
      cat("-> Bootstrap & Analytic CIs are consistent -> model appears stable.\n")
    }
  }

  # 保存推断结果
  saveRDS(inference_results, "../data/03_SCAD_Inference_Results.rds")

  # 推断报告
  cat("\n")
  cat("POST-SCAD STATISTICAL INFERENCE REPORT\n")

  cat("\n[1. Model Performance]\n")
  print(eval_metrics_df, row.names = FALSE)

  n_vars <- nrow(inference_results) - as.integer(has_intercept)
  n_scad_total <- attr(inference_results, "n_scad_selected")
  if (is.null(n_scad_total)) n_scad_total <- n_vars
  cat(sprintf("\n[2. Variable Selection]\n"))
  cat(sprintf("  SCAD selected         : %d features\n", n_scad_total))
  if (n_scad_total != n_vars) {
    cat(sprintf("  After instability removal: %d features\n", n_vars))
  }
  cat(sprintf("  Significant    : %d (at alpha = 0.05)\n",
              sum(sig_flag[inference_results$Variable != "(Intercept)"])))

  cat("\n[3. Coefficient Interpretation]\n")
  cat("  Coefficients are in log-odds scale.\n")
  cat("  Odds Ratio (OR) = exp(coef).  OR > 1 -> positive association.\n")
  cat("  A 95% CI that excludes 0 (log-odds) or 1 (OR) indicates significance.\n")

  glm_fit <- attr(inference_results, "glm_fit")
  cat(sprintf("\n[4. Refit Diagnostics]\n"))
  cat(sprintf("  Residual deviance : %.2f on %d df\n", deviance(glm_fit), df.residual(glm_fit)))
  cat(sprintf("  AIC               : %.2f\n", AIC(glm_fit)))

} else {
  cat("Inference calculation failed. Check model convergence and data quality.\n")
}

# 输出混淆矩阵并保存评估指标
cat("\n--- Confusion Matrix (Optimal Threshold) ---\n")
print(conf_matrix$table)
cat("\n")

save(prob_pred, roc_obj, pr_obj, conf_matrix, eval_metrics_df, Y_test_bin,
     inference_results, cv_scad,
     file = "../data/03_Evaluation_Metrics.RData")
