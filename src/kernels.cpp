#include "kernels.h"
#include <cmath>

namespace gp {

// ==================== RBF Kernel ====================

RBFKernel::RBFKernel(double lengthscale) {
  log_lengthscale = torch::nn::Parameter(torch::full({}, std::log(lengthscale)));
  register_parameter("log_lengthscale", log_lengthscale);
}

torch::Tensor RBFKernel::forward(const torch::Tensor& x1, const torch::Tensor& x2) {
  // x1: [n, d], x2: [m, d]
  // output: [n, m]
  
  // Compute squared distances
  auto x1_sq = (x1 * x1).sum(-1, true);  // [n, 1]
  auto x2_sq = (x2 * x2).sum(-1, true);  // [m, 1]
  auto cross = torch::matmul(x1, x2.t());  // [n, m]
  
  auto dist_sq = x1_sq + x2_sq.t() - 2 * cross;
  dist_sq = torch::clamp(dist_sq, 1e-8);  // Numerical stability
  
  auto lengthscale = torch::exp(log_lengthscale);
  return torch::exp(-dist_sq / (2 * lengthscale * lengthscale));
}

// ==================== Matern Kernel ====================

MaternKernel::MaternKernel(double lengthscale) {
  log_lengthscale = torch::nn::Parameter(torch::full({}, std::log(lengthscale)));
  register_parameter("log_lengthscale", log_lengthscale);
}

torch::Tensor MaternKernel::forward(const torch::Tensor& x1, const torch::Tensor& x2) {
  // Matern kernel with nu=2.5
  // K(d) = (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)
  
  auto x1_sq = (x1 * x1).sum(-1, true);  // [n, 1]
  auto x2_sq = (x2 * x2).sum(-1, true);  // [m, 1]
  auto cross = torch::matmul(x1, x2.t());  // [n, m]
  
  auto dist_sq = x1_sq + x2_sq.t() - 2 * cross;
  dist_sq = torch::clamp(dist_sq, 1e-8);
  auto dist = torch::sqrt(dist_sq);
  
  auto lengthscale = torch::exp(log_lengthscale);
  auto scaled_dist = dist / lengthscale;
  
  auto sqrt5 = std::sqrt(5.0);
  auto constant = 1.0 + sqrt5 * scaled_dist + (5.0 / 3.0) * scaled_dist * scaled_dist;
  auto exponential = torch::exp(-sqrt5 * scaled_dist);
  
  return constant * exponential;
}

// ==================== Scale Kernel ====================

ScaleKernel::ScaleKernel(std::shared_ptr<Kernel> base_kernel, double output_scale)
    : base_kernel(base_kernel) {
  log_output_scale = torch::nn::Parameter(torch::full({}, std::log(output_scale)));
  register_parameter("log_output_scale", log_output_scale);
}

torch::Tensor ScaleKernel::forward(const torch::Tensor& x1, const torch::Tensor& x2) {
  auto base_kernel_matrix = base_kernel->forward(x1, x2);
  auto output_scale = torch::exp(log_output_scale);
  return output_scale * base_kernel_matrix;
}

}  // namespace gp
