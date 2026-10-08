
# 前期数据读取与全量MDS
rm(list = ls())
gc()

source("GCor_SIS_REDS_MDS.R")
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
Sys.setenv(OMP_NUM_THREADS = 1)
Sys.setenv(MKL_NUM_THREADS = 1)

train_data <- readRDS("../data/02_Train_Ultrahigh.rds")
test_data  <- readRDS("../data/02_Test_Ultrahigh.rds")

# 强制剥离名字等属性，避免内存冗余复制
X_train <- unname(as.matrix(train_data$X))
X_test  <- unname(as.matrix(test_data$X))
Y_train <- unname(as.numeric(as.character(train_data$Y)))
Y_test  <- unname(as.numeric(as.character(test_data$Y)))

print("已读取数据，正在启动 GCor_SIS_REDS_MDS 筛选...")

# 执行全量MDS筛选
selected_results <- GCor_SIS_REDS_MDS(X_train, Y_train, 
                                        alpha = 0.1, 
                                        B = 20, 
                                        tau = 0.5)

### 【新增 1】：保存 SIS 阶段的核心结果 ###
saveRDS(selected_results, "../data/03_SIS_Results_Full.rds")
print("SIS阶段结果已保存至 03_SIS_Results_Full.rds")

# 提取最终筛选出的真实索引
selected_idx <- selected_results$final_selected
selection_freq <- selected_results$selection_freq

print(paste("最终筛选出的特征数量:", length(selected_idx)))
# print(paste("特征选择频率:", paste(round(selection_freq, 3), collapse = ", ")))


# 根据筛选出的索引提取子矩阵，供下一步 SCAD 模型使用
X_train_sis <- X_train[, selected_idx, drop = FALSE]
X_test_sis  <- X_test[, selected_idx, drop = FALSE]

# 确保 Y 必须是 0-1 因子或数值型 (1代表破产，0代表没破产)
Y_train_bin <- as.numeric(as.character(Y_train))
Y_test_bin  <- as.numeric(as.character(Y_test))
