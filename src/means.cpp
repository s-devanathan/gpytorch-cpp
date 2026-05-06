#include "means.h"

namespace gp {

// ==================== Constant Mean ====================

ConstantMean::ConstantMean(double constant_val) {
  constant = torch::nn::Parameter(torch::full({}, constant_val));
  register_parameter("constant", constant);
}

torch::Tensor ConstantMean::forward(const torch::Tensor& x) {
  // x: [n, d]
  // output: [n]
  return torch::full({x.size(0)}, constant.item<double>(), x.options());
}

// ==================== Zero Mean ====================

torch::Tensor ZeroMean::forward(const torch::Tensor& x) {
  // x: [n, d]
  // output: [n]
  return torch::zeros({x.size(0)}, x.options());
}

}  // namespace gp
