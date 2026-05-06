#include "gaussian_process.h"
#include <iostream>

namespace gp {

GaussianProcessRegressor::GaussianProcessRegressor(std::shared_ptr<Kernel> kernel,
                                                   std::shared_ptr<Mean> mean,
                                                   double noise_variance)
    : kernel_(kernel), mean_(mean) {
  log_noise_variance = torch::nn::Parameter(torch::full({}, std::log(noise_variance)));
  register_parameter("log_noise_variance", log_noise_variance);
}

void GaussianProcessRegressor::fit(const torch::Tensor& X_train,
                                   const torch::Tensor& y_train,
                                   int iterations,
                                   double learning_rate,
                                   bool verbose) {
  X_train_ = X_train.clone().detach().requires_grad_(false);
  y_train_ = y_train.clone().detach().requires_grad_(false);

  // Create optimizer
  torch::optim::Adam optimizer(parameters(), torch::optim::AdamOptions(learning_rate));

  for (int i = 0; i < iterations; ++i) {
    optimizer.zero_grad();

    // Compute log marginal likelihood
    auto lml = log_marginal_likelihood(X_train_, y_train_);
    auto loss = -lml;  // Minimize negative log marginal likelihood

    loss.backward();
    optimizer.step();

    if (verbose && (i + 1) % 10 == 0) {
      std::cout << "Iteration " << (i + 1) << " / " << iterations
                << " | Loss: " << loss.item<double>() << std::endl;
    }
  }

  // Cache the Cholesky decomposition for predictions
  update_kernel_cache();
}

torch::Tensor GaussianProcessRegressor::log_marginal_likelihood(const torch::Tensor& X,
                                                                const torch::Tensor& y) {
  // Compute kernel matrix
  auto K = kernel_->forward(X, X);
  auto noise_var = torch::exp(log_noise_variance);
  
  // Add noise: K + σ²I
  auto n = X.size(0);
  K = K + noise_var * torch::eye(n, X.options());

  // Cholesky decomposition
  auto L = torch::linalg_cholesky(K);

  // Compute y_centered = y - mean(X)
  auto mean_vals = mean_->forward(X);
  auto y_centered = y - mean_vals;

  // Compute α = K^{-1} y via Cholesky: L L^T α = y
  auto alpha = torch::cholesky_solve(y_centered.unsqueeze(-1), L).squeeze(-1);

  // Log marginal likelihood components
  auto data_fit = -0.5 * torch::dot(y_centered, alpha);
  auto complexity = -torch::sum(torch::log(torch::diagonal(L)));
  auto const_term = -0.5 * n * std::log(2 * M_PI);

  return data_fit + complexity + const_term;
}

void GaussianProcessRegressor::update_kernel_cache() {
  auto K = kernel_->forward(X_train_, X_train_);
  auto noise_var = torch::exp(log_noise_variance);
  auto n = X_train_.size(0);
  K = K + noise_var * torch::eye(n, X_train_.options());
  
  K_chol_ = torch::linalg_cholesky(K);
}

std::tuple<torch::Tensor, torch::Tensor> GaussianProcessRegressor::predict(
    const torch::Tensor& X_test) {
  torch::NoGradGuard no_grad;

  // Compute kernel matrices
  auto K_test_train = kernel_->forward(X_test, X_train_);  // [m, n]
  auto K_test_test = kernel_->forward(X_test, X_test);     // [m, m]
  
  // Get mean and noise variance
  auto mean_vals_train = mean_->forward(X_train_);
  auto mean_vals_test = mean_->forward(X_test);
  auto y_centered = y_train_ - mean_vals_train;

  // Solve K^{-1} K_test_train^T via Cholesky
  auto K_inv_K_test_t = torch::cholesky_solve(K_test_train.t().unsqueeze(-1), K_chol_);
  K_inv_K_test_t = K_inv_K_test_t.squeeze(-1);

  // Predictive mean: μ(x_*) = mean(x_*) + K(x_*, X) K^{-1} (y - mean(X))
  auto mean_pred = mean_vals_test + torch::matmul(K_test_train, 
                                                    torch::cholesky_solve(y_centered.unsqueeze(-1), K_chol_).squeeze(-1));

  // Predictive variance: σ²(x_*) = K(x_*, x_*) - K(x_*, X) K^{-1} K(X, x_*)
  auto var_pred = K_test_test - torch::matmul(K_test_train, K_inv_K_test_t.t());
  var_pred = torch::clamp(torch::diagonal(var_pred, 0, -2, -1), 1e-8);

  return std::make_tuple(mean_pred, var_pred);
}

}  // namespace gp
