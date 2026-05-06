#pragma once

#include <torch/torch.h>
#include "kernels.h"
#include "means.h"
#include <memory>
#include <tuple>

namespace gp {

class GaussianProcessRegressor {
 public:
  GaussianProcessRegressor(std::shared_ptr<Kernel> kernel,
                           std::shared_ptr<Mean> mean,
                           double noise_variance = 0.01);

  // Fit GP to training data
  void fit(const torch::Tensor& X_train, 
           const torch::Tensor& y_train,
           int iterations = 100,
           double learning_rate = 0.01,
           bool verbose = true);

  // Make predictions
  std::tuple<torch::Tensor, torch::Tensor> predict(const torch::Tensor& X_test);

  // Get log marginal likelihood
  torch::Tensor log_marginal_likelihood(const torch::Tensor& X, const torch::Tensor& y);

 private:
  std::shared_ptr<Kernel> kernel_;
  std::shared_ptr<Mean> mean_;
  
  torch::Tensor X_train_;
  torch::Tensor y_train_;
  torch::Tensor K_chol_;  // Cholesky decomposition of K + σ²I
  
  torch::nn::Parameter log_noise_variance;

  // Compute Cholesky decomposition and cache training data
  void update_kernel_cache();
  
  // Compute squared euclidean distance between points
  torch::Tensor squared_distance(const torch::Tensor& x1, const torch::Tensor& x2);
};

}  // namespace gp
