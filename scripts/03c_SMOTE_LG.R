# 载入必要的包
library(smotefamily) # 用于 SMOTE
library(pROC)
library(PRROC)
library(ROCR)
library(caret)
library(ggplot2)
library(patchwork)   # 用于拼接图像

theme_colors <- c("#c6c077", "#8aae76", "#59977a", "#377d79", "#2d636e", "#2f4858")

# 1. 数据加载与预处理
train_data <- readRDS("../data/01_Train_Data.rds")
test_data  <- readRDS("../data/01_Test_Data.rds")

# 逻辑回归需要 data.frame 格式，而不是 matrix
X_train <- as.data.frame(train_data$X)
X_test  <- as.data.frame(test_data$X)
Y_train <- as.numeric(as.character(train_data$Y))
Y_test  <- as.numeric(as.character(test_data$Y))

# 2. SMOTE 过采样处理
set.seed(42) # 保证过采样结果的可重复性
# 注意：smotefamily 会将特征和目标变量合并，并将目标变量命名为 "class"
smote_obj <- SMOTE(X = X_train, target = Y_train)
smote_train_data <- smote_obj$data

# 将生成的 target ('class') 转换为 factor 类型以适应 glm 的二分类
smote_train_data$class <- as.factor(smote_train_data$class)

# 3. 逻辑回归模型训练
# 警告：若特征间存在严重多重共线性，基础 glm 可能不收敛。若发生此情况请改用 glmnet。
lr_model <- glm(class ~ ., data = smote_train_data, family = binomial(link = "logit"))

# 4. 核心统计指标计算
# 使用模型对测试集进行概率预测
preds_prob_lr <- predict(lr_model, newdata = X_test, type = "response")

# 处理可能存在的 NA 预测值（当测试集中存在训练集中未出现的 factor level 或特征时）
preds_prob_lr[is.na(preds_prob_lr)] <- 0 

# ROC 与阈值寻优
roc_obj_lr <- roc(Y_test, preds_prob_lr, quiet = TRUE)
auc_value_lr <- auc(roc_obj_lr)
best_coords_lr <- coords(roc_obj_lr, "best", ret = c("threshold", "specificity", "sensitivity"))
opt_thresh_lr <- best_coords_lr$threshold[1]

# PR-AUC 计算
fg_lr <- preds_prob_lr[Y_test == 1]
bg_lr <- preds_prob_lr[Y_test == 0]
pr_obj_lr <- pr.curve(scores.class0 = fg_lr, scores.class1 = bg_lr, curve = TRUE)
pr_auc_value_lr <- pr_obj_lr$auc.integral

# KS 统计量计算
clean_preds_lr <- as.numeric(preds_prob_lr)
clean_Y_test <- as.numeric(Y_test)
pred_rocr_lr <- prediction(clean_preds_lr, clean_Y_test)
perf_tpr_lr <- performance(pred_rocr_lr, "tpr")
perf_fpr_lr <- performance(pred_rocr_lr, "fpr")

cutoffs_lr <- perf_tpr_lr@x.values[[1]]
tpr_vals_lr <- perf_tpr_lr@y.values[[1]]
fpr_vals_lr <- perf_fpr_lr@y.values[[1]]

ks_diff_lr <- tpr_vals_lr - fpr_vals_lr
ks_stat_lr <- max(ks_diff_lr, na.rm = TRUE)
ks_index_lr <- which.max(ks_diff_lr)

# 混淆矩阵与分类指标
preds_class_lr <- factor(ifelse(preds_prob_lr >= opt_thresh_lr, 1, 0), levels = c(0, 1))
Y_test_factor <- factor(Y_test, levels = c(0, 1))
conf_matrix_lr <- confusionMatrix(preds_class_lr, Y_test_factor, positive = "1")

precision_lr <- conf_matrix_lr$byClass["Pos Pred Value"]
recall_lr <- conf_matrix_lr$byClass["Sensitivity"]
f1_score_lr <- conf_matrix_lr$byClass["F1"]

if(is.na(precision_lr)) precision_lr <- 0
if(is.na(recall_lr)) recall_lr <- 0

# F2-Score 计算
f2_score_lr <- (1 + 2^2) * (precision_lr * recall_lr) / ((2^2 * precision_lr) + recall_lr)
if(is.na(f2_score_lr)) f2_score_lr <- 0

# 5. 统一风格的可视化绘制
# 5.1 ROC Curve
roc_df_lr <- data.frame(FPR = 1 - roc_obj_lr$specificities, TPR = roc_obj_lr$sensitivities)
p_roc_lr <- ggplot(roc_df_lr, aes(x = FPR, y = TPR)) +
  geom_line(color = theme_colors[4], linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "ROC Curve (Logistic Regression)", subtitle = sprintf("AUC = %.4f", auc_value_lr), 
       x = "False Positive Rate", y = "True Positive Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 5.2 PR Curve
pr_df_lr <- data.frame(Recall = pr_obj_lr$curve[, 1], Precision = pr_obj_lr$curve[, 2])
p_pr_lr <- ggplot(pr_df_lr, aes(x = Recall, y = Precision)) +
  geom_line(color = theme_colors[5], linewidth = 1.2) +
  theme_minimal(base_size = 14) +
  labs(title = "Precision-Recall Curve (LR)", subtitle = sprintf("PR-AUC = %.4f", pr_auc_value_lr),
       x = "Recall (Sensitivity)", y = "Precision (PPV)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 5.3 KS Curve
ks_df_lr <- data.frame(Threshold = cutoffs_lr, TPR = tpr_vals_lr, FPR = fpr_vals_lr)
p_ks_lr <- ggplot(ks_df_lr, aes(x = Threshold)) +
  geom_line(aes(y = TPR, color = "TPR"), linewidth = 1.2) +
  geom_line(aes(y = FPR, color = "FPR"), linewidth = 1.2) +
  geom_vline(xintercept = cutoffs_lr[ks_index_lr], linetype = "dashed", color = "black") +
  scale_color_manual(name = "Metrics", values = c("TPR" = theme_colors[4], "FPR" = theme_colors[2])) +
  theme_minimal(base_size = 14) +
  labs(title = "KS Curve (LR)", subtitle = sprintf("KS Statistic = %.4f", ks_stat_lr),
       x = "Probability Threshold", y = "Cumulative Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15), legend.position = "bottom")

# 5.4 Calibration Curve
calib_data_lr <- data.frame(
  obs = factor(ifelse(Y_test == 1, "Bankruptcy", "Healthy"), levels = c("Bankruptcy", "Healthy")),
  Bankruptcy = preds_prob_lr
)
calib_obj_lr <- calibration(obs ~ Bankruptcy, data = calib_data_lr, class = "Bankruptcy", cuts = 10)
calib_df_lr <- calib_obj_lr$data

p_calib_lr <- ggplot(calib_df_lr, aes(x = midpoint, y = Percent)) +
  geom_line(color = theme_colors[6], linewidth = 1.2) +
  geom_point(color = theme_colors[6], size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "Calibration Curve (LR)", 
       x = "Bin Midpoint (Predicted Probability %)", y = "Observed Event Percentage (%)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 输出所有图像 (不含 SHAP，因为基础 LR 系数即为特征解释)
p_combined_lr <- (p_roc_lr | p_pr_lr) / 
  (p_ks_lr | p_calib_lr) +
  plot_annotation(
    title = "Logistic Regression (SMOTE) Evaluation Metrics",
    theme = theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5))
  )
print(p_combined_lr)

# 6. 结果汇总与集中输出
eval_metrics_df_lr <- data.frame(
  Metric = c("ROC-AUC", "PR-AUC", "KS Statistic", "Optimal Threshold", 
             "Precision (PPV)", "Recall (Sensitivity)", "F1-Score", "F2-Score"),
  Value = c(auc_value_lr, pr_auc_value_lr, ks_stat_lr, opt_thresh_lr, 
            precision_lr, recall_lr, f1_score_lr, f2_score_lr)
)

eval_metrics_df_lr$Value <- round(eval_metrics_df_lr$Value, 4)

cat("\nLogistic Regression (SMOTE) Evaluation Metrics:\n")
print(eval_metrics_df_lr, row.names = FALSE)

cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix_lr$table)

# 保存 LR 结果
save(lr_model, preds_prob_lr, roc_obj_lr, pr_obj_lr, conf_matrix_lr, eval_metrics_df_lr,
     file = "../data/04_LogisticRegression_Evaluation.RData")