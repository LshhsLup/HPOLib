#include <gtest/gtest.h>
#include <tensor/Half.h>
#include <iostream>
#include <limits>
#include <vector>

using namespace hpolib;

#if defined(__CUDACC__)
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                << ": " << cudaGetErrorString(err) << std::endl;           \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)
#endif

TEST(HalfCPUTest, ConstructionAndConversion) {
  EXPECT_NEAR(static_cast<float>(Half(1.0f)), 1.0f, 1e-3);
  EXPECT_NEAR(static_cast<float>(Half(-3.5)), -3.5f, 1e-3);
  EXPECT_NEAR(static_cast<float>(Half(100)), 100.0f, 1e-3);
  EXPECT_EQ(static_cast<int>(Half(123.45f)), 123);
}

TEST(HalfCPUTest, UserDefinedLiterals) {
  auto h1 = 1.23_hf;
  auto h2 = 10_hf;
  EXPECT_EQ(h1, Half(1.23f));
  EXPECT_EQ(h2, Half(10.0f));
}

TEST(HalfCPUTest, ArithmeticOperators) {
  Half a = 2.5_hf;
  Half b = 4.0_hf;

  EXPECT_EQ(a + b, 6.5_hf);
  EXPECT_EQ(a - b, -1.5_hf);
  EXPECT_EQ(a * b, 10.0_hf);
  EXPECT_EQ(b / a, 1.6_hf);
  EXPECT_EQ(-a, -2.5_hf);

  a += b;  // a is now 6.5
  EXPECT_EQ(a, 6.5_hf);
  b *= 2.0_hf;  // b is now 8.0
  EXPECT_EQ(b, 8.0_hf);
}

TEST(HalfCPUTest, ComparisonOperators) {
  Half a = 5.0_hf;
  Half b = 5.0_hf;
  Half c = 6.0_hf;

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a < c);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(c > a);
  EXPECT_TRUE(c >= b);

  Half nan_val = std::numeric_limits<Half>::quiet_NaN();
  EXPECT_FALSE(nan_val == nan_val);
  EXPECT_TRUE(nan_val != nan_val);
}

TEST(HalfCPUTest, SpecialValuesAndFunctions) {
  Half inf = std::numeric_limits<Half>::infinity();
  Half neg_inf = -inf;
  Half nan = std::numeric_limits<Half>::quiet_NaN();
  Half zero = 0.0_hf;
  Half neg_zero = -0.0_hf;

  EXPECT_TRUE(isinf(inf));
  EXPECT_TRUE(isinf(neg_inf));
  EXPECT_FALSE(isfinite(inf));
  EXPECT_TRUE(isnan(nan));

  EXPECT_EQ(fpclassify(inf), FP_INFINITE);
  EXPECT_EQ(fpclassify(nan), FP_NAN);
  EXPECT_EQ(fpclassify(1.0_hf), FP_NORMAL);
  EXPECT_EQ(fpclassify(zero), FP_ZERO);

  EXPECT_EQ(abs(-5.5_hf), 5.5_hf);
  EXPECT_TRUE(isinf(abs(neg_inf)));

  EXPECT_EQ(copysign(10.0_hf, -1.0_hf), -10.0_hf);
  EXPECT_EQ(copysign(-10.0_hf, 1.0_hf), 10.0_hf);

  EXPECT_FALSE(signbit(1.0_hf));
  EXPECT_TRUE(signbit(-1.0_hf));
  EXPECT_FALSE(signbit(zero));
  EXPECT_TRUE(signbit(neg_zero));
}

TEST(HalfCPUTest, NumericLimits) {
  EXPECT_TRUE(std::numeric_limits<Half>::is_specialized);
  EXPECT_EQ(std::numeric_limits<Half>::infinity(),
            Half(std::numeric_limits<float>::infinity()));
  EXPECT_TRUE(isnan(std::numeric_limits<Half>::quiet_NaN()));

  EXPECT_EQ(static_cast<float>(std::numeric_limits<Half>::max()), 65504.0f);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<Half>::lowest()), -65504.0f);
}

#if defined(__CUDACC__)
__global__ void comprehensive_half_kernel(const Half* a, const Half* b, Half* c,
                                          char* results, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // 测试算术
    Half val_a = a[idx];
    Half val_b = b[idx];
    c[idx] = (val_a + val_b) * 2.0_hf;

    // 测试比较
    results[idx] = (val_a < val_b);
  }
}

TEST(HalfGPUTest, ComprehensiveOperations) {
  const int n = 512;
  const size_t half_size = n * sizeof(Half);
  const size_t result_size = n * sizeof(char);

  std::vector<Half> h_a(n);
  std::vector<Half> h_b(n);
  std::vector<Half> h_c(n);
  std::vector<char> h_results(n);

  for (int i = 0; i < n; ++i) {
    h_a[i] = Half(static_cast<float>(i));
    h_b[i] = Half(static_cast<float>(n - i));
  }

  Half *d_a, *d_b, *d_c;
  char* d_results;
  CUDA_CHECK(cudaMalloc(&d_a, half_size));
  CUDA_CHECK(cudaMalloc(&d_b, half_size));
  CUDA_CHECK(cudaMalloc(&d_c, half_size));
  CUDA_CHECK(cudaMalloc(&d_results, result_size));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), half_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), half_size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  comprehensive_half_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c,
                                                                d_results, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, half_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, result_size,
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    float expected_c =
        (static_cast<float>(i) + static_cast<float>(n - i)) * 2.0f;
    EXPECT_NEAR(static_cast<float>(h_c[i]), expected_c, 0.001f);

    bool expected_result = (static_cast<float>(i) < static_cast<float>(n - i));
    EXPECT_EQ(static_cast<bool>(h_results[i]), expected_result);
  }
  std::cout << "GPU comprehensive test validation passed for " << n
            << " elements." << std::endl;

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_results));
}
#endif
