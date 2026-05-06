# Gaussian Process Regressor in C++

A simple Gaussian Process implementation in C++ using LibTorch, inspired by GPyTorch's design patterns.

## Features

- **Multiple Kernels**: RBF, Matern, and ScaleKernel
- **Mean Functions**: Constant and Zero mean
- **Hyperparameter Optimization**: Uses Adam optimizer to learn kernel and noise parameters
- **Efficient Predictions**: Cholesky decomposition for numerical stability
- **PyTorch Integration**: Built on LibTorch with automatic differentiation

## Architecture

```
include/
├── kernels.h              # Kernel definitions (RBF, Matern, Scale)
├── means.h                # Mean function definitions
└── gaussian_process.h     # Main GP regressor

src/
├── kernels.cpp            # Kernel implementations
├── means.cpp              # Mean function implementations
└── gaussian_process.cpp   # GP regressor implementation

examples/
└── simple_gp.cpp          # Usage example
```

## Building

### Prerequisites
- C++17 or later
- CMake 3.15+
- PyTorch/LibTorch (CPU or GPU)

### Linux/macOS

```bash
# Install PyTorch (via conda recommended)
conda install pytorch::pytorch -c pytorch

# Export LibTorch path
export TORCH_HOME=/path/to/pytorch/install

# Build
mkdir build && cd build
cmake ..
make
```

### Running Example

```bash
./gp_example
```

## Usage Example

```cpp
#include "gaussian_process.h"
#include "kernels.h"
#include "means.h"

// Generate training data
torch::Tensor X_train = ...; // [n_samples, n_features]
torch::Tensor y_train = ...; // [n_samples]

// Create GP with RBF kernel
auto kernel = std::make_shared<gp::RBFKernel>(1.0);
auto mean = std::make_shared<gp::ZeroMean>();
auto gp = std::make_shared<gp::GaussianProcessRegressor>(kernel, mean);

// Fit to training data
gp->fit(X_train, y_train, iterations=100, learning_rate=0.01);

// Make predictions
torch::Tensor X_test = ...; // [m_samples, n_features]
auto [mean_pred, var_pred] = gp->predict(X_test);
```

## Key Components

### Kernels

- **RBFKernel**: Radial Basis Function kernel with learnable lengthscale
- **MaternKernel**: Matern kernel with nu=2.5 (smoother than RBF)
- **ScaleKernel**: Wraps any kernel and adds an output scale parameter

### Mean Functions

- **ConstantMean**: Learnable constant mean
- **ZeroMean**: Zero mean function

### Gaussian Process Regressor

The main `GaussianProcessRegressor` class provides:

1. **fit()**: Optimizes hyperparameters using negative log marginal likelihood
2. **predict()**: Returns predictive mean and variance
3. **Log Marginal Likelihood**: Computed via Cholesky decomposition

## Algorithm Details

### Prediction

Given training data {(x_i, y_i)} and test points x_*:

```
μ(x_*) = K(x_*, X) [K(X, X) + σ²I]^{-1} y
σ²(x_*) = K(x_*, x_*) - K(x_*, X) [K(X, X) + σ²I]^{-1} K(X, x_*)
```

### Hyperparameter Learning

Hyperparameters are optimized by maximizing the log marginal likelihood:

```
log p(y|X) = -0.5 y^T K^{-1} y - 0.5 log|K| - n/2 log(2π)
```

Computed efficiently using Cholesky decomposition.

## Differences from GPyTorch

- Simplified to demonstrate core concepts
- Single-task regression only
- No inducing points (exact GP)
- Limited to PyTorch/LibTorch backend
- No GPU-optimized routines (though LibTorch supports them)

## Performance Considerations

- Time complexity for prediction: O(n²m) where n=training samples, m=test samples
- Space complexity: O(n²) for kernel matrices
- For larger datasets, consider inducing point approximations

## Future Extensions

- Sparse GPs with inducing points
- Multi-task GPs
- Variational inference
- GPU optimization
- Neural process variants

## License

MIT
