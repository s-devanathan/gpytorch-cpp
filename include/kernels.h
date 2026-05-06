#pragma once

#include <torch/torch.h>
#include <memory>

namespace gp {

// Base Kernel class
class Kernel : public torch::nn::Module {
 public:
  virtual ~Kernel() = default;
  virtual torch::Tensor forward(const torch::Tensor& x1, const torch::Tensor& x2) = 0;
};

// RBF (Squared Exponential) Kernel
class RBFKernel : public Kernel {
 public:
  explicit RBFKernel(double lengthscale = 1.0);
  torch::Tensor forward(const torch::Tensor& x1, const torch::Tensor& x2) override;
  
 private:
  torch::nn::Parameter log_lengthscale;
};

// Matern Kernel (nu=2.5)
class MaternKernel : public Kernel {
 public:
  explicit MaternKernel(double lengthscale = 1.0);
  torch::Tensor forward(const torch::Tensor& x1, const torch::Tensor& x2) override;
  
 private:
  torch::nn::Parameter log_lengthscale;
};

// Scale Kernel (wraps another kernel with output scale)
class ScaleKernel : public Kernel {
 public:
  explicit ScaleKernel(std::shared_ptr<Kernel> base_kernel, double output_scale = 1.0);
  torch::Tensor forward(const torch::Tensor& x1, const torch::Tensor& x2) override;
  
 private:
  std::shared_ptr<Kernel> base_kernel;
  torch::nn::Parameter log_output_scale;
};

}  // namespace gp
