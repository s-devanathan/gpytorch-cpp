#include <iostream>
#include <vector>
#include "gaussian_process.h"
#include "kernels.h"
#include "means.h"

int main() {
  std::cout << "=== Gaussian Process Regressor Example ===" << std::endl;

  // Generate synthetic data: y = sin(x) + noise
  int n_train = 20;
  auto X_train = torch::linspace(0, 2 * M_PI, n_train).unsqueeze(-1);
  auto y_train = torch::sin(X_train).squeeze() + 0.1 * torch::randn({n_train});

  std::cout << "\nTraining data:" << std::endl;
  std::cout << "  X shape: " << X_train.sizes() << std::endl;
  std::cout << "  y shape: " << y_train.sizes() << std::endl;

  // Create GP with RBF kernel
  auto kernel = std::make_shared<gp::RBFKernel>(1.0);
  auto mean = std::make_shared<gp::ZeroMean>();
  auto gp = std::make_shared<gp::GaussianProcessRegressor>(kernel, mean, 0.01);

  // Fit the GP
  std::cout << "\nFitting GP..." << std::endl;
  gp->fit(X_train, y_train, 50, 0.1, true);

  // Make predictions
  std::cout << "\nMaking predictions..." << std::endl;
  int n_test = 100;
  auto X_test = torch::linspace(-0.5, 2 * M_PI + 0.5, n_test).unsqueeze(-1);
  auto [mean_pred, var_pred] = gp->predict(X_test);

  std::cout << "Predictions:" << std::endl;
  std::cout << "  Mean shape: " << mean_pred.sizes() << std::endl;
  std::cout << "  Variance shape: " << var_pred.sizes() << std::endl;

  // Print some predictions
  std::cout << "\nSample predictions (first 5):" << std::endl;
  for (int i = 0; i < 5; ++i) {
    auto x_val = X_test[i].item<double>();
    auto mean_val = mean_pred[i].item<double>();
    auto std_val = std::sqrt(var_pred[i].item<double>());
    std::cout << "  x=" << x_val << " | μ=" << mean_val << " | σ=" << std_val << std::endl;
  }

  std::cout << "\n=== Complete ===" << std::endl;
  return 0;
}
