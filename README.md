# CoreForge
HPOLib 是一个致力于高性能计算（HPC）和并行计算的算子库项目。其核心目标是为各种数值计算和深度学习算子提供优化的 CPU 单线程、CPU 多线程（OpenMP/TBB）以及 CUDA (GPU) 实现。通过这个项目，旨在深入理解不同计算平台上的并行编程范式、性能优化技巧，并构建一个功能丰富且高效的算子集合。

## 支持的算子
目前计划支持以下算子。此列表将根据项目进展持续更新和扩展：
基本数学运算: Abs, AddN, Div, Exp, Log, Pow, Sqrt, Square, Reciprocal, Round, Trunc, Ceil, Floor, Sin, Cos, Tan, Atan2, Angle, ComplexAbs, Conj, Xlogy, Polygamma, Erf, Rsqrt, NegTensor, AdvancedIndex
激活函数: ActivationForward/Backward (ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Hardtanh, Prelu)
张量操作: Concat, Split, Slice, Transpose, Tile, Expand, Roll, Flip, Pad, UnsortedSegmentSum, Where, OneHot, Arange, Linspace, TopKTensor, Gather, Scatter, BatchGatherV2, ScatterNd, ScatterRef, DynamicPartition, DynamicStitch, RepeatInterleave, ShuffleChannel, AsStrided, CropAndResize
矩阵与线性代数: MatMul, BatchMatMul/BCast, ComplexMatMul, Det, Inverse, QR, Cross, Trace, Diag/Diagonal/DiagPart, Tri/TriIndices
卷积与池化: Conv/Deconv Forward/Backward, AdaptivePoolingForward/Backward, FractionalMaxPoolForward, Im2Col/Col2Im, DCNForward/Backward (Deformable Convolution)
归一化: BatchNormForward/Backward, FrozenBatchNormBackward, LayerNormForward/Backward, InstanceNormForward/Backward, GroupNormForward/Backward, Lrn/Grad
损失函数: BceLoss/Backward, MSELoss/Backward, NlllossForward/Backward, CTCLoss
优化器相关: ApplyAdam, Clip/GradNorm
深度学习模块: Embedding/Bag Forward/Backward, GRU/GRUCell Forward/Backward, LSTM/LSTMGates Forward/Backward, BiasAdd/Backward, FusedDropout, Softmax/Backward
其他: BitCompute, BoxOverlapBev, Bucketize, CastDataType, CosineSimilarity, GridSampleForward/Backward, Histc, Interp/Backward, Lerp, LogicalOp/Not, Maximum/Minimum/Median, MulN, Nms, Normalize/Backward, OpTensor, PointsInBoxes, Quantize/Param, RandomUniform/Normal, Reduce, Rotate, Space2batch/Nd, Cummax/min/prod/sum

```
HPOLib
├─ .clang-format
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ csrc
│  ├─ common
│  │  └─ cpu_info.h
│  ├─ core
│  │  ├─ BFloat16.h
│  │  ├─ DType.h
│  │  ├─ Half.h
│  │  ├─ allocator.h
│  │  ├─ config.h
│  │  ├─ cuda.h
│  │  ├─ device.h
│  │  ├─ logger.h
│  │  ├─ options.h
│  │  ├─ scalar.h
│  │  ├─ storage.h
│  │  ├─ tensor.h
│  │  ├─ tensorImpl.h
│  │  ├─ utils.h
│  │  └─ vectorUtils.h
│  └─ kernels
│     ├─ CMakeLists.txt
│     ├─ abs
│     │  ├─ cpu
│     │  │  ├─ abs.cpp
│     │  │  └─ abs.h
│     │  └─ cuda
│     │     ├─ abs.cu
│     │     └─ abs.cuh
│     ├─ activation_backward
│     └─ activation_forward
├─ format_code.sh
└─ tests
   ├─ CMakeLists.txt
   └─ tensor
      └─ test_half.cu

```
```
HPOLib
├─ .clang-format
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ csrc
│  ├─ common
│  │  └─ cpu_info.h
│  ├─ core
│  │  ├─ BFloat16.h
│  │  ├─ DType.h
│  │  ├─ Half.h
│  │  ├─ allocator.h
│  │  ├─ config.h
│  │  ├─ cuda.h
│  │  ├─ device.h
│  │  ├─ logger.h
│  │  ├─ options.h
│  │  ├─ scalar.h
│  │  ├─ storage.h
│  │  ├─ tensor.h
│  │  ├─ tensorImpl.h
│  │  ├─ utils.h
│  │  └─ vectorUtils.h
│  └─ kernels
│     ├─ CMakeLists.txt
│     ├─ abs
│     │  ├─ abs.h
│     │  ├─ abs_cpu.cpp
│     │  └─ abs_cuda.cu
│     ├─ activation_backward
│     └─ activation_forward
├─ format_code.sh
└─ tests
   ├─ CMakeLists.txt
   └─ tensor
      └─ test_half.cu

```
```
HPOLib
├─ .clang-format
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ csrc
│  ├─ common
│  │  └─ cpu_info.h
│  ├─ core
│  │  ├─ BFloat16.h
│  │  ├─ DType.h
│  │  ├─ Half.h
│  │  ├─ allocator.h
│  │  ├─ config.h
│  │  ├─ cuda.h
│  │  ├─ device.h
│  │  ├─ logger.h
│  │  ├─ options.h
│  │  ├─ scalar.h
│  │  ├─ storage.h
│  │  ├─ tensor.h
│  │  ├─ tensorImpl.h
│  │  ├─ utils.h
│  │  └─ vectorUtils.h
│  ├─ kernels
│  │  ├─ CMakeLists.txt
│  │  ├─ abs
│  │  │  ├─ abs.h
│  │  │  ├─ abs_cpu.cpp
│  │  │  └─ abs_cuda.cu
│  │  ├─ activation_backward
│  │  └─ activation_forward
│  └─ ops
│     ├─ abs.cpp
│     └─ api.h
├─ format_code.sh
└─ tests
   ├─ CMakeLists.txt
   └─ tensor
      └─ test_half.cu

```
```
HPOLib
├─ .clang-format
├─ CMakeLists.txt
├─ LICENSE
├─ README.md
├─ csrc
│  ├─ common
│  │  └─ cpu_info.h
│  ├─ core
│  │  ├─ BFloat16.h
│  │  ├─ DType.h
│  │  ├─ Half.h
│  │  ├─ allocator.h
│  │  ├─ config.h
│  │  ├─ cuda.h
│  │  ├─ device.h
│  │  ├─ logger.h
│  │  ├─ options.h
│  │  ├─ scalar.h
│  │  ├─ storage.h
│  │  ├─ tensor.h
│  │  ├─ tensorImpl.h
│  │  ├─ utils.h
│  │  └─ vectorUtils.h
│  ├─ kernels
│  │  ├─ CMakeLists.txt
│  │  ├─ abs
│  │  │  ├─ abs.h
│  │  │  ├─ abs_cpu.cpp
│  │  │  └─ abs_cuda.cu
│  │  ├─ activation_backward
│  │  └─ activation_forward
│  └─ ops
│     ├─ abs.cpp
│     └─ api.h
├─ format_code.sh
└─ tests
   ├─ CMakeLists.txt
   └─ tensor
      └─ test_half.cu

```