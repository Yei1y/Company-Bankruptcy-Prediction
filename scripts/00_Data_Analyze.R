# 脚本功能：破产检测数据集探索性数据分析
library(ggplot2)
library(dplyr)
library(tidyr)

theme_colors <- c("#c6c077", "#8aae76", "#59977a", "#377d79", "#2d636e", "#2f4858")
color_stable <- theme_colors[2]   # 财务稳健 (0)
color_bankrupt <- theme_colors[1] # 破产 (1)

# 读取数据
raw_data <- read.csv("../data/data.csv")
data_matrix <- as.matrix(raw_data)
data_matrix[is.nan(data_matrix)] <- NA
data <- na.omit(data_matrix)

# 提取特征与目标变量
Y_train <- data[, 1]
X_train <- as.data.frame(data[, -1])

# 目标变量转为因子
target_status <- factor(
  ifelse(Y_train == 1, "Bankrupt", "Financially Stable"),
  levels = c("Financially Stable", "Bankrupt")
)

plot_data <- data.frame(Status = target_status, X_train)

# 各经济学意义区块的特征索引
idx_profitability <- c(1:12, 15, 42, 43, 68, 69, 86, 89, 90, 94)
idx_leverage      <- c(14, 35:41, 60, 64, 66, 67, 77:79, 84, 85, 91:93, 95)
idx_cashflow      <- c(13, 20, 32:34, 54:59, 61:63, 65, 74, 75, 80:83)
idx_efficiency    <- c(44:53, 70:73, 76, 87, 88)
idx_growth        <- c(16:19, 21:31)

# 计算全局相关性热力图
library(ggplot2)
library(dplyr)
library(tidyr)

# 构造全量数据矩阵，目标变量命名为 Y，特征用数字索引
full_data_matrix <- cbind(Y = Y_train, X_train)
colnames(full_data_matrix) <- c("Y", as.character(1:ncol(X_train)))

# 计算相关系数矩阵
global_cormat <- cor(full_data_matrix, use = "pairwise.complete.obs")

# 提取各特征与 Y 的绝对相关性
abs_pb_corr <- abs(global_cormat[-1, "Y"])
names(abs_pb_corr) <- colnames(X_train)

# 按与 Y 的相关性排序，聚集共线性模块
target_corr_vector <- global_cormat[, "Y"]
ordered_vars <- names(sort(target_corr_vector, decreasing = TRUE))
ordered_cormat <- global_cormat[ordered_vars, ordered_vars]

# 转长格式并锁定因子顺序
cormat_df_global <- as.data.frame(as.table(ordered_cormat))
colnames(cormat_df_global) <- c("Var1", "Var2", "Correlation")
cormat_df_global <- cormat_df_global %>%
  mutate(Var1 = factor(Var1, levels = ordered_vars),
         Var2 = factor(Var2, levels = rev(ordered_vars)))

# 绘制全景热力图
p_global_heatmap <- ggplot(cormat_df_global, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_raster() +
  scale_fill_gradient2(low = theme_colors[5], high = theme_colors[1], mid = "#f7f7f7",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Pearson\nCorrelation") +
  theme_minimal(base_size = 14) +
  labs(title = "Global Correlation Heatmap",
       subtitle = "Features represented by indices, sorted by correlation with Y",
       x = "Feature Index / Y", y = "Feature Index / Y") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
        axis.text.y = element_text(size = 6),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        legend.position = "right") +
  coord_fixed()

print(p_global_heatmap)

# 用 GAM 平滑曲线探测核心指标的非线性关系
# 抽取第一个盈利能力指标作为示例特征
sample_feature <- X_train[, idx_profitability[1]]
gam_plot_data <- data.frame(Target = Y_train, Feature = sample_feature)

p_nonlinear <- ggplot(gam_plot_data, aes(x = Feature, y = Target)) +
  geom_jitter(aes(color = factor(Target)), height = 0.05, alpha = 0.3, size = 1.5) +
  # 使用受罚三次回归样条 (cs) 拟合广义相加模型，评估非线性展开的必要性
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"),
              color = theme_colors[5], fill = theme_colors[4], alpha = 0.2, linewidth = 1.2) +
  scale_color_manual(values = c("0" = color_stable, "1" = color_bankrupt)) +
  theme_minimal(base_size = 15) +
  labs(title = "Non-linear Relationship Exploration",
       subtitle = "GAM smoothing curve justifying cubic spline expansion",
       x = paste("Feature Index:", idx_profitability[1]), y = "Probability of Bankruptcy") +
  theme(legend.position = "none")

print(p_nonlinear)

# 绘制各经济区块特征的边缘预测效用
idx_growth <- c(16:19, 21:31)

marginal_utility_df <- data.frame(
  Feature = colnames(X_train),
  Abs_Corr = abs_pb_corr,
  Block = NA
)

marginal_utility_df$Block[idx_profitability] <- "Profitability"
marginal_utility_df$Block[idx_leverage]      <- "Leverage"
marginal_utility_df$Block[idx_cashflow]      <- "Cashflow"
marginal_utility_df$Block[idx_efficiency]    <- "Efficiency"
marginal_utility_df$Block[idx_growth]        <- "Growth"

# 剔除未分类的特征
marginal_utility_df <- na.omit(marginal_utility_df)

p_marginal <- ggplot(marginal_utility_df, aes(x = reorder(Block, Abs_Corr, FUN = median), y = Abs_Corr, fill = Block)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6, color = theme_colors[6]) +
  scale_fill_manual(values = c("Profitability" = theme_colors[1],
                               "Leverage" = theme_colors[2],
                               "Cashflow" = theme_colors[3],
                               "Efficiency" = theme_colors[4],
                               "Growth" = theme_colors[5])) +
  theme_minimal(base_size = 15) +
  labs(title = "Marginal Utility of Features by Economic Block",
       subtitle = "Absolute point-biserial correlation with Target Y (Updated with Growth)",
       x = "Economic Block", y = "Absolute Correlation with Status") +
  theme(legend.position = "none")

print(p_marginal)

# 拟合各经济区块的多元逻辑回归并评估效用
library(patchwork)

# 定义核心建模与绘图函数
fit_and_plot_block <- function(block_name, feature_indices) {

  # 准备该区块的子数据集
  block_data <- data.frame(Target = Y_train, X_train[, feature_indices])

  # 拟合逻辑回归模型
  fit <- glm(Target ~ ., data = block_data, family = binomial(link = "logit"))

  # 计算 McFadden 伪 R^2
  null_model <- glm(Target ~ 1, data = block_data, family = binomial(link = "logit"))
  ll_fit <- as.numeric(logLik(fit))
  ll_null <- as.numeric(logLik(null_model))
  pseudo_r2 <- 1 - (ll_fit / ll_null)

  # 提取预测概率与线性预测子
  # 用线性组合得分 (Logit) 作 X 轴，降维展示多元拟合效果
  plot_df <- data.frame(
    Actual = factor(Y_train, levels = c(0, 1), labels = c("Financially Stable", "Bankrupt")),
    Predicted_Prob = fit$fitted.values,
    Linear_Predictor = fit$linear.predictors
  )

  # 绘制拟合图
  p <- ggplot(plot_df, aes(x = Linear_Predictor, y = Predicted_Prob)) +
    geom_point(aes(color = Actual), alpha = 0.5, size = 1.5) +
    stat_function(fun = function(x) 1 / (1 + exp(-x)), color = "black", linewidth = 0.8) +
    scale_color_manual(values = c("Financially Stable" = theme_colors[2], "Bankrupt" = theme_colors[1])) +
    theme_minimal(base_size = 12) +
    labs(title = block_name,
         x = "Linear Predictor (Log-odds)",
         y = "Predicted Probability") +
    annotate("text", x = min(plot_df$Linear_Predictor), y = 0.95,
             label = sprintf("McFadden R^2: %.3f", pseudo_r2),
             hjust = 0, fontface = "italic", size = 4.5) +
    theme(legend.position = "none")

  return(p)
}

# 组合 5 个区块的拟合结果
# 压制完全分离问题导致的 GLM 警告
suppressWarnings({
  p_profit     <- fit_and_plot_block("Profitability", idx_profitability)
  p_leverage   <- fit_and_plot_block("Leverage", idx_leverage)
  p_cashflow   <- fit_and_plot_block("Cashflow", idx_cashflow)
  p_efficiency <- fit_and_plot_block("Efficiency", idx_efficiency)
  p_growth     <- fit_and_plot_block("Growth", idx_growth)
})

# 用 patchwork 构建 3 行 2 列布局，右下角留空
combined_plot <- (p_profit | p_leverage) /
  (p_cashflow | p_efficiency) /
  (p_growth | plot_spacer()) +
  plot_annotation(
    title = "Multivariate Logistic Regression Fits by Economic Block",
    subtitle = "Sigmoid curves mapped to linear predictors with McFadden's Pseudo-R² (5 Blocks)",
    theme = theme(plot.title = element_text(size = 16, face = "bold"))
  )

print(combined_plot)

# 基于统计显著性提取 Top 核心特征做分布与相关性展示
# 原始特征 95 维，全局绘图会引发视觉过载

# 绘制核心特征的类别条件分布直方图
# 提取相关性最强的前 9 个特征
top_9_features <- names(sort(abs_pb_corr, decreasing = TRUE))[1:9]

hist_data <- plot_data %>%
  select(Status, all_of(top_9_features)) %>%
  pivot_longer(cols = -Status, names_to = "Feature", values_to = "Value") %>%
  mutate(Feature = factor(Feature, levels = top_9_features))

p_hist <- ggplot(hist_data, aes(x = Value, fill = Status)) +
  # 用 position = "identity" 透明叠加，评估两类样本在单变量上的密度重叠度
  geom_histogram(alpha = 0.55, position = "identity", bins = 40, color = "white", linewidth = 0.2) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c("Financially Stable" = color_stable, "Bankrupt" = color_bankrupt)) +
  theme_minimal(base_size = 14) +
  labs(title = "Class-Conditional Distributions of Top Predictive Features",
       subtitle = "Overlaid histograms displaying feature separability for bankruptcy status",
       x = "Normalized Feature Value", y = "Count") +
  theme(legend.position = "top",
        strip.text = element_text(face = "bold", size = 9),
        panel.grid.minor = element_blank(),
        panel.spacing = unit(1.2, "lines"))

print(p_hist)

# 绘制核心变量群的相关性热力图
# 提取相关性最强的前 15 个特征以及目标变量 Y
top_15_features <- names(sort(abs_pb_corr, decreasing = TRUE))[1:15]
corr_data_matrix <- cbind(Target = Y_train, X_train[, top_15_features])

# 计算变量间的 Pearson 相关系数矩阵
cormat <- cor(corr_data_matrix, use = "pairwise.complete.obs")

# 重构矩阵为长格式数据框
cormat_df <- as.data.frame(cormat)
cormat_df$Var1 <- rownames(cormat)
cormat_df <- cormat_df %>%
  pivot_longer(cols = -Var1, names_to = "Var2", values_to = "Correlation") %>%
  # 控制因子水平，使对角线从左上到右下排列
  mutate(Var1 = factor(Var1, levels = rownames(cormat)),
         Var2 = factor(Var2, levels = rev(rownames(cormat))))

p_heatmap <- ggplot(cormat_df, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile(color = "white", linewidth = 0.6) +
  scale_fill_gradient2(low = theme_colors[5], high = theme_colors[1], mid = "#f7f7f7",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Pearson\nCorrelation") +
  geom_text(aes(label = ifelse(abs(Correlation) >= 1.1, sprintf("%.2f", Correlation), "")),
            color = "black", size = 3.2, fontface = "bold") +
  theme_minimal(base_size = 13) +
  labs(title = "Correlation Matrix of Target and Top 15 Features",
       subtitle = "Visualizing multicollinearity and marginal predictive utility",
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 9),
        axis.text.y = element_text(size = 9),
        panel.grid = element_blank(),
        legend.position = "right") +
  coord_fixed()

print(p_heatmap)

# 计算组内与组间平均绝对相关系数
# 评估各经济区块内部的信息聚合度与区块间的信息独立性
cat("\n[INFO] 开始计算经济学区块的组内与组间平均相关系数...\n")

block_list <- list(
  Profitability = idx_profitability,
  Leverage      = idx_leverage,
  Cashflow      = idx_cashflow,
  Efficiency    = idx_efficiency,
  Growth        = idx_growth
)

block_names <- names(block_list)
n_blocks <- length(block_names)

block_cor_matrix <- matrix(NA, nrow = n_blocks, ncol = n_blocks,
                           dimnames = list(block_names, block_names))

# 计算全体特征的相关系数矩阵
cor_X <- cor(X_train, use = "pairwise.complete.obs")

for (i in 1:n_blocks) {
  for (j in 1:n_blocks) {
    idx_i <- block_list[[i]]
    idx_j <- block_list[[j]]

    if (i == j) {
      # 组内平均：提取上三角部分，忽略零方差特征产生的 NA
      sub_cor <- abs(cor_X[idx_i, idx_i])
      block_cor_matrix[i, j] <- mean(sub_cor[upper.tri(sub_cor)], na.rm = TRUE)
    } else {
      # 组间平均
      cross_cor <- abs(cor_X[idx_i, idx_j])
      block_cor_matrix[i, j] <- mean(cross_cor, na.rm = TRUE)
    }
  }
}

# 提取宏观统计量时加入 na.rm = TRUE
global_intra_avg <- mean(diag(block_cor_matrix), na.rm = TRUE)
global_inter_avg <- mean(block_cor_matrix[upper.tri(block_cor_matrix)], na.rm = TRUE)

# 打印规范化报告
print(round(block_cor_matrix, 4))
cat(sprintf("=> 全局平均组内相关系数: %.4f\n", global_intra_avg))
cat(sprintf("=> 全局平均组间相关系数: %.4f\n", global_inter_avg))


# 绘制区块级平均绝对相关系数矩阵热力图
# 转换矩阵为长格式数据框
block_cor_df <- as.data.frame(as.table(block_cor_matrix))
colnames(block_cor_df) <- c("Block1", "Block2", "Avg_Abs_Corr")

# 锁定因子顺序确保对角线方向
block_levels <- rownames(block_cor_matrix)
block_cor_df <- block_cor_df %>%
  mutate(Block1 = factor(Block1, levels = block_levels),
         Block2 = factor(Block2, levels = rev(block_levels)))

p_block_heatmap <- ggplot(block_cor_df, aes(x = Block1, y = Block2, fill = Avg_Abs_Corr)) +
  geom_tile(color = "white", linewidth = 1) +
  scale_fill_gradient(low = "#f7f7f7", high = theme_colors[1],
                      na.value = "grey80", name = "Mean Abs\nCorrelation") +
  geom_text(aes(label = sprintf("%.4f", Avg_Abs_Corr)),
            color = "black", size = 4.5, fontface = "bold") +
  theme_minimal(base_size = 14) +
  labs(title = "Block-level Average Absolute Correlation Matrix",
       subtitle = "Diagonal: Intra-block (Expected High) | Off-diagonal: Inter-block (Expected Low)",
       x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
        axis.text.y = element_text(size = 11, face = "bold"),
        panel.grid = element_blank(),
        plot.title = element_text(face = "bold", size = 15)) +
  coord_fixed()

print(p_block_heatmap)
