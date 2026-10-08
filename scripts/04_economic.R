# 脚本功能：基于反向概率校准的 SCAD 模型经济效益评估
# 还原违约概率、计算经济学最优阈值、绘制校准曲线与利润曲线
library(ggplot2)
library(dplyr)
library(ncvreg)
library(caret)
library(patchwork)

theme_colors <- c("#c6c077", "#8aae76", "#59977a", "#377d79", "#2d636e", "#2f4858")

# 加载数据与 SCAD 模型
cat("[1/4] 加载原始数据与 SCAD 模型...\n")
sis_results <- readRDS("../data/03_SIS_Results_Full.rds")
train_data  <- readRDS("../data/02_Train_Ultrahigh.rds")
test_data   <- readRDS("../data/02_Test_Ultrahigh.rds")
cv_scad     <- readRDS("../data/03_SCAD_Model.rds")

# 提取特征
idx <- which(sis_results$selection_freq > 0.7)
X_test_sis  <- as.matrix(test_data$X)[, idx, drop = FALSE]
Y_train_bin <- as.numeric(as.character(train_data$Y))
Y_test_bin  <- as.numeric(as.character(test_data$Y))

# 基于样本比例做反向概率校准
cat("[2/4] 执行反向校准以还原真实概率...\n")

# 获取最优 lambda 下的原始系数
raw_coefs <- coef(cv_scad, lambda = cv_scad$lambda.min)

# 计算线性预测器 X*beta
# X_test_sis 无常数列，单独加截距项
linear_predictor <- raw_coefs[1] + (X_test_sis %*% raw_coefs[-1])

# 计算 Sigmoid 预测概率 p_weighted
p_weighted <- as.vector(1 / (1 + exp(-linear_predictor)))

# 应用反向校准公式还原物理概率
prob_pred_real <- p_weighted / (p_weighted + 32 * (1 - p_weighted))

# 测算经济场景与预期利润
cat("[3/4] 测算经济场景与预期利润...\n")

evaluate_economic_profit <- function(prob, Y_true, r, LGD, exposure = 1000000) {
  optimal_p <- r / (r + LGD)
  decision_to_lend <- prob < optimal_p

  loans_approved <- sum(decision_to_lend)
  success_loans <- sum(decision_to_lend == TRUE & Y_true == 0)
  default_loans <- sum(decision_to_lend == TRUE & Y_true == 1)

  revenue <- success_loans * (exposure * r)
  loss <- default_loans * (exposure * LGD)
  net_profit <- revenue - loss

  return(data.frame(
    Scenario = paste0("Interest: ", r*100, "%, LGD: ", LGD*100, "%"),
    Optimal_Threshold = round(optimal_p, 4),
    Approved = loans_approved,
    Defaults = default_loans,
    Net_Profit = net_profit
  ))
}

# 测算信用贷场景 (高风险, LGD=100%, 利率=5%)
scene_A <- evaluate_economic_profit(prob_pred_real, Y_test_bin, r = 0.05, LGD = 1.00)
# 测算抵押贷场景 (低风险, LGD=40%, 利率=6%)
scene_B <- evaluate_economic_profit(prob_pred_real, Y_test_bin, r = 0.06, LGD = 0.40)

economic_report <- rbind(scene_A, scene_B)
economic_report$Formatted_Profit <- paste0("¥", format(economic_report$Net_Profit, big.mark = ",", scientific = FALSE))
print(economic_report[, c("Scenario", "Optimal_Threshold", "Approved", "Defaults", "Formatted_Profit")], row.names = FALSE)

# 绘制校准曲线与经济利润曲线
cat("[4/4] 生成修正后的校准曲线与经济利润曲线...\n")

# 绘制等频校准曲线 (Decile Calibration Curve)
calib_df <- data.frame(Predicted = prob_pred_real, Observed = Y_test_bin)
calib_df <- calib_df %>%
  mutate(Bin = ntile(Predicted, 10)) %>%
  group_by(Bin) %>%
  summarise(
    Mean_Predicted = mean(Predicted) * 100,
    Event_Rate = mean(Observed) * 100
  )

p_calib_real <- ggplot(calib_df, aes(x = Mean_Predicted, y = Event_Rate)) +
  geom_line(color = theme_colors[4], linewidth = 1.2) +
  geom_point(color = theme_colors[5], size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  theme_minimal(base_size = 14) +
  labs(title = "Adjusted Calibration Curve (Decile Bins)",
       subtitle = "Probabilities calibrated via odds-ratio adjustment",
       x = "Mean Predicted Probability (%)", y = "Observed Event Rate (%)") +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold", size = 15))

# 绘制经济期望利润曲线 (Profit Curve)
scan_thresholds <- seq(0, 0.20, by = 0.001)
profits_A <- sapply(scan_thresholds, function(t) {
  lend <- prob_pred_real < t
  (sum(lend & Y_test_bin == 0) * 1000000 * 0.05) - (sum(lend & Y_test_bin == 1) * 1000000 * 1.0)
})
profits_B <- sapply(scan_thresholds, function(t) {
  lend <- prob_pred_real < t
  (sum(lend & Y_test_bin == 0) * 1000000 * 0.06) - (sum(lend & Y_test_bin == 1) * 1000000 * 0.4)
})

plot_profit_df <- data.frame(
  Threshold = rep(scan_thresholds, 2),
  Profit = c(profits_A, profits_B),
  Scenario = rep(c("Scenario A (r=5%, LGD=100%)", "Scenario B (r=6%, LGD=40%)"), each = length(scan_thresholds))
)

p_profit <- ggplot(plot_profit_df, aes(x = Threshold, y = Profit, color = Scenario)) +
  geom_line(linewidth = 1.2) +
  geom_vline(xintercept = scene_A$Optimal_Threshold, linetype = "dashed", color = theme_colors[3]) +
  geom_vline(xintercept = scene_B$Optimal_Threshold, linetype = "dashed", color = theme_colors[6]) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  scale_y_continuous(labels = scales::label_number(scale = 1e-6, suffix = "M", prefix = "¥")) +
  scale_color_manual(values = c(theme_colors[3], theme_colors[6])) +
  theme_minimal(base_size = 14) +
  labs(title = "Expected Total Profit by Decision Threshold",
       subtitle = "Dashed lines indicate theoretical optimal p*",
       x = "Real Probability Decision Threshold (Cut-off)", y = "Total Profit (Millions)") +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold", size = 15))

# 组合出图
p_final <- p_calib_real + p_profit + plot_layout(widths = c(1, 1.5))
print(p_final)

cat("\n脚本执行完毕，校准曲线与利润曲线已生成。\n")
