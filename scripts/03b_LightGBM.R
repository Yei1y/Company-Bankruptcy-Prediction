library(lightgbm)
library(pROC)
library(PRROC)
library(ROCR)
library(caret)
library(shapviz)
library(ggplot2)

theme_colors <- c("#c6c077", "#8aae76", "#59977a", "#377d79", "#2d636e", "#2f4858")

# 1. 数据加载与 Dataset 构建
train_data <- readRDS("../data/01_Train_Data.rds")
test_data  <- readRDS("../data/01_Test_Data.rds")

X_train <- as.matrix(train_data$X)
X_test  <- as.matrix(test_data$X)
Y_train <- as.numeric(as.character(train_data$Y))
Y_test  <- as.numeric(as.character(test_data$Y))

dtrain <- lgb.Dataset(data = X_train, label = Y_train)
dtest  <- lgb.Dataset(data = X_test,  label = Y_test, reference = dtrain)

# 2. 计算不平衡权重与模型训练
pos_weight <- sum(Y_train == 0) / sum(Y_train == 1)

params <- list(
  objective = "binary",
  metric = c("auc", "binary_logloss"),
  learning_rate = 0.05,
  num_leaves = 31,
  max_depth = 6,
  scale_pos_weight = pos_weight,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  min_data_in_leaf = 20
)

model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  valids = list(train = dtrain, test = dtest),
  early_stopping_rounds = 50,
  verbose = -1 
)

# 3. 核心统计指标计算
preds_prob <- predict(model, X_test)

# ROC 与阈值寻优
roc_obj <- roc(Y_test, preds_prob, quiet = TRUE)
auc_value <- auc(roc_obj)
best_coords <- coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
opt_thresh <- best_coords$threshold[1]

# PR-AUC 计算
fg <- preds_prob[Y_test == 1]
bg <- preds_prob[Y_test == 0]
pr_obj <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
pr_auc_value <- pr_obj$auc.integral

# KS 统计量计算
pred_rocr <- prediction(preds_prob, Y_test)
perf_tpr <- performance(pred_rocr, "tpr")
perf_fpr <- performance(pred_rocr, "fpr")

cutoffs <- perf_tpr@x.values[[1]]
tpr_vals <- perf_tpr@y.values[[1]]
fpr_vals <- perf_fpr@y.values[[1]]

ks_diff <- tpr_vals - fpr_vals
ks_stat <- max(ks_diff)
ks_index <- which.max(ks_diff)

# 混淆矩阵与分类指标
preds_class <- factor(ifelse(preds_prob >= opt_thresh, 1, 0), levels = c(0, 1))
Y_test_factor <- factor(Y_test, levels = c(0, 1))
conf_matrix <- confusionMatrix(preds_class, Y_test_factor, positive = "1")

precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f1_score <- conf_matrix$byClass["F1"]

if(is.na(precision)) precision <- 0
if(is.na(recall)) recall <- 0

# F2-Score 计算
f2_score <- (1 + 2^2) * (precision * recall) / ((2^2 * precision) + recall)
if(is.na(f2_score)) f2_score <- 0

# 4. 统一风格的可视化绘制
# 4.1 ROC Curve
roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = theme_colors[4], linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  theme_minimal(base_size = 14) +
  labs(title = "ROC Curve", subtitle = sprintf("AUC = %.4f", auc_value), 
       x = "False Positive Rate", y = "True Positive Rate") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 4.2 PR Curve
pr_df <- data.frame(Recall = pr_obj$curve[, 1], Precision = pr_obj$curve[, 2])
p_pr <- ggplot(pr_df, aes(x = Recall, y = Precision)) +
  geom_line(color = theme_colors[5], linewidth = 1.2) +
  theme_minimal(base_size = 14) +
  labs(title = "Precision-Recall Curve", subtitle = sprintf("PR-AUC = %.4f", pr_auc_value),
       x = "Recall (Sensitivity)", y = "Precision (PPV)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 4.3 KS Curve
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

# 4.4 Calibration Curve
calib_data <- data.frame(
  obs = factor(ifelse(Y_test == 1, "Bankruptcy", "Healthy"), levels = c("Bankruptcy", "Healthy")),
  Bankruptcy = preds_prob
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

# 4.5 SHAP Beeswarm
shp <- shapviz(model, X_pred = X_test)
p_shap <- sv_importance(shp, kind = "beeswarm") +
  theme_minimal(base_size = 14) +
  scale_color_gradient(low = theme_colors[2], high = theme_colors[6]) +
  labs(title = "SHAP Beeswarm: Non-linear Impact of Features") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 输出所有图像
p_combined <- (p_roc | p_pr) / 
  (p_ks | p_calib) +
  plot_annotation(
    title = "Comprehensive Model Evaluation Metrics",
    theme = theme(plot.title = element_text(face = "bold", size = 18, hjust = 0.5))
  )
# 输出组合图
print(p_combined)
print(p_shap)

# 计算特征重要性矩阵
importance_matrix <- lgb.importance(model, percentage = TRUE)

# lgb.importance 返回的行数即为发生过分裂的有效特征数量
effective_feature_count <- nrow(importance_matrix)

cat("\nEffective Features Count (Based on Splits/Gain):", effective_feature_count, "\n")

# 5. 结果汇总与集中输出
eval_metrics_df <- data.frame(
  Metric = c("ROC-AUC", "PR-AUC", "KS Statistic", "Optimal Threshold", 
             "Precision (PPV)", "Recall (Sensitivity)", "F1-Score", "F2-Score"),
  Value = c(auc_value, pr_auc_value, ks_stat, opt_thresh, 
            precision, recall, f1_score, f2_score)
)

eval_metrics_df$Value <- round(eval_metrics_df$Value, 4)

cat("\nModel Evaluation Metrics:\n")
print(eval_metrics_df, row.names = FALSE)

cat("\nConfusion Matrix (Optimal Threshold):\n")
print(conf_matrix$table)

save(model, preds_prob, roc_obj, pr_obj, conf_matrix, eval_metrics_df,
     file = "../data/04_LightGBM_Evaluation.RData")