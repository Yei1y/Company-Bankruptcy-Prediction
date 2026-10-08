cpp_source_code <- '
#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// 直接传矩阵和列索引，避免反复创建 Rcpp 对象
void compute_kernel_matrix_optimized(const NumericMatrix& X, int col_idx, NumericMatrix& A, NumericVector& norms, int n) {
  for(int i = 0; i < n; i++) {
      norms[i] = sqrt(X(i, col_idx) * X(i, col_idx) + 1.0);
  }
  double val, ratio;
  for(int i = 0; i < n; i++) {
    for(int j = 0; j <= i; j++) {
        val = X(i, col_idx) * X(j, col_idx) + 1.0;
        ratio = val / (norms[i] * norms[j]);
        if(ratio > 1.0) ratio = 1.0;
        if(ratio < -1.0) ratio = -1.0;
        double asin_val = std::asin(ratio);
        A(i, j) = asin_val;
        A(j, i) = asin_val;
    }
  }
}

void compute_kernel_matrix_Y(const NumericVector& y, NumericMatrix& A, NumericVector& norms, int n) {
  for(int i = 0; i < n; i++) norms[i] = sqrt(y[i]*y[i] + 1.0);
  double val, ratio;
  for(int i = 0; i < n; i++) {
    for(int j = 0; j <= i; j++) {
        val = y[i]*y[j] + 1.0;
        ratio = val / (norms[i] * norms[j]);
        if(ratio > 1.0) ratio = 1.0;
        if(ratio < -1.0) ratio = -1.0;
        double asin_val = std::asin(ratio);
        A(i, j) = asin_val;
        A(j, i) = asin_val;
    }
  }
}

double center_and_sum_sq(NumericMatrix& A, int n, NumericVector& row_sums) {
  for(int i = 0; i < n; i++) A(i, i) = 0.0;
  double total_sum = 0.0;
  for(int i = 0; i < n; i++) {
    double r_sum = 0;
    for(int j = 0; j < n; j++) r_sum += A(i, j);
    row_sums[i] = r_sum;
    total_sum += r_sum;
  }
  double grand_mean = total_sum / ((n - 1.0) * (n - 2.0));
  double sq_sum = 0.0;
  for(int i = 0; i < n; i++) {
    double mean_i = row_sums[i] / (n - 2.0);
    for(int j = 0; j < n; j++) {
        if (i == j) { A(i, j) = 0.0; }
        else {
            double mean_j = row_sums[j] / (n - 2.0);
            double centered_val = A(i, j) - mean_i - mean_j + grand_mean;
            A(i, j) = centered_val;
            sq_sum += centered_val * centered_val;
        }
    }
  }
  return sq_sum;
}

// [[Rcpp::export]]
NumericVector GCor_SIS_cpp_loop(NumericMatrix X, NumericVector Y) {
    int p = X.ncol();
    int n = X.nrow();
    NumericVector results(p);

    // 缓冲区只分配一次
    NumericMatrix Mat_Buffer_Y(n, n);
    NumericMatrix Mat_Buffer_X(n, n);
    NumericVector Norms_Buffer(n);
    NumericVector RowSum_Buffer(n);

    compute_kernel_matrix_Y(Y, Mat_Buffer_Y, Norms_Buffer, n);
    double Y_sq_sum = center_and_sum_sq(Mat_Buffer_Y, n, RowSum_Buffer);
    if (Y_sq_sum == 0) return results;

    for(int j = 0; j < p; j++) {
        // 传列索引而非创建 X_col 对象，避免触发 R GC
        compute_kernel_matrix_optimized(X, j, Mat_Buffer_X, Norms_Buffer, n);
        double X_sq_sum = center_and_sum_sq(Mat_Buffer_X, n, RowSum_Buffer);
        if (X_sq_sum == 0) { results[j] = 0.0; }
        else {
            double num = 0.0;
           for(int i = 0; i < n * n; i++) { num += Mat_Buffer_X[i] * Mat_Buffer_Y[i]; }
           double cor = num / sqrt(X_sq_sum * Y_sq_sum);
           results[j] = cor * cor;
        }
    }
    return results;
}
'

# 核心单次筛选算法
GCor_SIS_REDS <- function(X, Y, alpha = 0.05) {
  n <- nrow(X)
  p <- ncol(X)

  # 划分 D1 和 D2 的索引
  n1 <- floor(2 * n / 3)
  n2 <- n - n1
  idx1 <- sample(1:n, n1)

  # 内存优化：算完即销毁，逐段加载到内存
  # 阶段 A：用 D1 计算
  X1 <- X[idx1, , drop = FALSE]
  Y1 <- Y[idx1]
  omega1 <- GCor_SIS_cpp_loop(X1, Y1)
  rm(X1, Y1)
  gc()

  # 阶段 B：用 D2 计算
  X2 <- X[-idx1, , drop = FALSE]
  Y2 <- Y[-idx1]
  omega2 <- GCor_SIS_cpp_loop(X2, Y2)
  rm(X2, Y2)
  gc()

  # 计算 M_r (公式 4)
  term1 <- n1 * omega1
  term2 <- n2 * omega2
  M <- sign(term1 - term2) * pmax(term1, term2)

  # 寻找 t_opt (公式 6)
  t_candidates <- sort(unique(abs(M[M != 0])))
  t_opt <- Inf

  for (t in t_candidates) {
    numerator <- 1 + sum(M <= -t)
    denominator <- max(sum(M >= t), 1)

    if ((numerator / denominator) <= alpha) {
      t_opt <- t
      break
    }
  }

  # 获取活跃预测变量集
  if (is.infinite(t_opt)) {
    active_predictors <- integer(0)
  } else {
    active_predictors <- which(M >= t_opt)
  }

  return(list(
    selected = active_predictors,
    t_opt = t_opt,
    M_values = M
  ))
}

# 全量多重拆分算法
library(doParallel)
library(foreach)

GCor_SIS_REDS_MDS <- function(X, Y, alpha = 0.1, B = 20, tau = 0.5) {

  # 转为纯矩阵/向量，避免 Rcpp 因数据框属性额外复制大矩阵
  if (!is.matrix(X)) X <- unname(as.matrix(X))
  if (!is.numeric(Y)) Y <- unname(as.numeric(Y))

  p <- ncol(X)

  # 解除 R 核心数限制，预留 2 核给系统
  Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
  n_cores <- 20

  cat(sprintf("启动纯并行全量 MDS，使用核心数: %d，共执行 %d 次拆分...\n", n_cores, B))

  cl <- makeCluster(n_cores)

  # 向工作节点分发 C++ 代码
  clusterExport(cl, "cpp_source_code", envir = .GlobalEnv)
  clusterEvalQ(cl, {
    library(Rcpp)
    cppFunction(cpp_source_code)
  })

  registerDoParallel(cl)

  start_time <- Sys.time()

  # 执行并行多重拆分
  results_list <- foreach(b = 1:B,
                          .packages = c("Rcpp"),
                          .export = c("GCor_SIS_REDS")) %dopar% {

    res_single <- GCor_SIS_REDS(X, Y, alpha = alpha)
    return(res_single$selected)
  }

  stopCluster(cl)
  end_time <- Sys.time()

  cat(sprintf("全量并行计算完成！耗时: %.2f 秒\n", as.numeric(difftime(end_time, start_time, units="secs"))))

  # 汇总选择次数
  selection_counts <- numeric(p)
  for (selected_vars in results_list) {
    if (length(selected_vars) > 0) {
      selection_counts[selected_vars] <- selection_counts[selected_vars] + 1
    }
  }

  selection_freq <- selection_counts / B
  final_selected <- which(selection_freq >= tau)

  cat(sprintf("全流程结束。最终从 %d 维全样本中严选出 %d 个特征。\n", p, length(final_selected)))
  return(list(
    final_selected = final_selected,
    selection_freq = selection_freq
  ))
}
