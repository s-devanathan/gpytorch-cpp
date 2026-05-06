#pragma once

#include <torch/torch.h>

namespace gp {

// Base Mean class
class Mean : public torch::nn::Module {
 public:
  virtual ~Mean() = default;
  virtual torch::Tensor forward(const torch::Tensor& x) = 0;
};

// Constant Mean Function
class ConstantMean : public Mean {
 public:
  explicit ConstantMean(double constant = 0.0);
  torch::Tensor forward(const torch::Tensor& x) override;
  
 private:
  torch::nn::Parameter constant;
};

// Zero Mean Function
class ZeroMean : public Mean {
 public:
  ZeroMean() = default;
  torch::Tensor forward(const torch::Tensor& x) override;
};

}  // namespace gp
