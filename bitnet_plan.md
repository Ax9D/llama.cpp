# 1. BitNet Kernel Architecture and Mobile GPU Execution Strategy

## Overview

This project targets GPU acceleration of BitNet inference on mobile platforms including:

- Apple GPUs (Metal)
- Qualcomm Adreno GPUs (Vulkan)
- ARM Mali GPUs (Vulkan)

The implementation will focus on extending the existing TQ1_0 infrastructure already present in GGML and llama.cpp rather than introducing a completely new quantization format.

---

# 2. llama.cpp Integration and Cross-Platform Backend Extension Plan

llama.cpp already contains support for BitNet model inference (`models/bitnet.cpp`), however the current implementation is CPU-only.

## Quantization Scheme

The TQ1_0 quantization format represents weights using ternary values ("trits"). During quantization, weights within a block are first centered around the block's average value and then mapped to the set `{-1, 0, +1}` (see `quantize_row_tq1_0_ref` in `ggml/src/ggml-quants.c`). During dequantization, the block scale factor must be applied to reconstruct the original weight values.

For storage, the ternary values are remapped from `{-1, 0, +1}` to `{0, 1, 2}`. Multiple trits are packed into a single byte by interpreting them as a base-3 number.

Five weights are packed into a byte because:

- `3^5 = 243 < 256`
- `3^6 = 729 > 256`

and therefore six trits cannot fit in a single byte.

## Block Layout

```c
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5];
    uint8_t qh[QK_K / 64];
    ggml_half d;
} block_tq1_0;
```

Each block stores 256 weights (`QK_K`) distributed across three regions.

### Section 1 (`qs`)

- First 160 weights
- Stored in 32 bytes
- 5 trits per byte

### Section 2 (`qs + 32`)

- Next 80 weights
- Stored in 16 bytes
- 5 trits per byte

### Section 3 (`qh`)

- Final 16 weights
- Stored in 4-byte chunks
- 4 weights per byte

## Weight Layout

Weights are stored in a strided pattern:

```text
Byte 0: w0   w32  w64  w96  w128
Byte 1: w1   w33  w65  w97  w129
Byte 2: w2   w34  w66  w98  w130
...
```

As a result, dequantization cannot simply walk through consecutive trits. The kernel must traverse bytes and extract the correct trit according to the packing stride.

## Current TQ1_0 Support

- CPU implementations for TQ1_0 matrix multiplication
- Optimized Vulkan mat-vec kernels
- Optimized Vulkan mat-mat kernels
- Optimized Metal mat-vec kernels
- Optimized Metal mat-mat kernels
- On-the-fly dequantization during GPU execution
- GGUF model loading support
- Tensor storage support

## Missing Functionality

### TQ1_0 Dequantization Support for Mat-Mat Kernels

Specialized dequantization paths are required for matrix-matrix multiplication.

### TQ1_0 Quantized Dot Product Support

Support for TQ1_0 quantized dot products is required within matrix-vector kernels.

These kernels directly impact:

- Prompt processing (prefill)
- Token generation (decode)

---

# 3. Performance Profiling Methodology and Optimization Workflow

`llama-bench` will be used as the primary benchmarking tool.

The initial objective will be to compare:

- Prompt processing throughput (pp)
- Token generation throughput (tg)

against existing GGML quantization formats.

Profiling will then be performed using vendor-specific GPU tooling to identify bottlenecks such as:

- Trit decoding overhead
- Memory bandwidth utilization
- Occupancy limitations
- Synchronization overhead
- Register pressure

The profiling process will guide architecture-specific optimization efforts across Apple, Adreno, and Mali GPUs.

---

# 4. Development Environment, Frameworks, Libraries, and SDK Requirements

The implementation will be developed primarily in C++ and integrated into the existing llama.cpp codebase.

Rather than building entirely new inference backends, the work will focus on extending and optimizing existing TQ1_0 support already present in GGML.

## Core Frameworks

- llama.cpp
- GGML
- GGUF

## Graphics and Compute APIs

### Vulkan

Target platforms:

- Qualcomm Adreno
- ARM Mali

### Metal

Target platforms:

- Apple GPUs

## Development Toolchain

- C++17 / C++20
- CMake
- Clang / LLVM
- Git

## Platform SDKs

### Android

- Android SDK
- Android NDK
- Vulkan SDK

### Apple

- Xcode
- Metal Framework
- Metal Shader Compiler

## Profiling and Performance Analysis Tools

### Apple GPUs

- Xcode GPU Trace

### Qualcomm Adreno GPUs

- RenderDoc
- Snapdragon Profiler

### ARM Mali GPUs

- ARM Streamline

## Validation and Benchmarking

- llama.cpp `test-backend-ops`
- llama-bench

---

# 5. Kernel-Level Optimization Techniques

The primary goal is to maximize the benefits of BitNet's ternary representation while minimizing decoding overhead and memory bandwidth consumption.

## On-the-Fly Weight Decoding

Weights will remain in their compressed ternary representation and be decoded directly inside matrix multiplication kernels.

Benefits:

- Reduced memory footprint
- Lower memory bandwidth requirements
- Better cache utilization

## BitNet-Aware Tiling

Matrix multiplication kernels will process tiles of activations and packed ternary weights. Decoded weights will be consumed immediately rather than expanded into temporary buffers.

## Exploiting Weight Packing Structure

The implementation will:

- Decode multiple trits per load
- Reuse decoded values across accumulations
- Minimize expensive division and modulo operations
- Exploit the existing strided TQ1_0 packing layout

## Subgroup and SIMD Utilization

The implementation will leverage:

- Vulkan subgroup operations
- Metal SIMD-group operations

for efficient reductions and accumulation.

## Shared Memory / Threadgroup Memory Usage

Activation tiles will be staged in:

- Shared memory (Vulkan)
- Threadgroup memory (Metal)

to reduce global memory traffic.

## Register-Level Accumulation

Accumulation will remain in registers whenever possible to maximize arithmetic intensity.

## Workgroup and Tile Size Tuning

Tile sizes and workgroup configurations will be tuned separately for:

- Apple GPUs
- Qualcomm Adreno GPUs
- ARM Mali GPUs

---

### Hardware Matrix Accelerators

The existing GGML Vulkan and Metal backends already make use of hardware matrix acceleration where available through:

- Metal tensor and SIMD-accelerated matrix multiplication paths
- Vulkan cooperative matrix extensions (`VK_KHR_cooperative_matrix`)

As part of this work, the existing TQ1_0 kernels will be extended and profiled to determine whether BitNet inference can effectively benefit from these hardware acceleration paths.

A key consideration is that BitNet/TQ1_0 weights require ternary decoding before participating in matrix multiplication. Therefore, profiling will be performed to evaluate:

- The cost of ternary weight decoding relative to matrix multiplication.
- Whether decoded weight tiles can be efficiently consumed by existing cooperative matrix implementations.
- The performance and energy efficiency benefits of hardware matrix acceleration for TQ1_0 workloads.

The results of this evaluation will guide whether additional TQ1_0-specific optimizations are required around the existing hardware-accelerated matrix multiplication kernels.

# 6. Evaluation Methodology, Success Criteria, and Benchmarking Strategy

## Correctness Validation

Validation will include:

- Verification of ternary weight decoding
- Validation of matrix-vector outputs
- Validation of matrix-matrix outputs
- Comparison against GGML CPU implementations
- Numerical error analysis

### Metrics

- Mean Absolute Error (MAE)
- Maximum Absolute Error
- Relative Error

### Success Criteria

- Outputs match CPU reference implementations within acceptable tolerance
- No accuracy regressions relative to existing TQ1_0 execution paths

## Kernel-Level Performance Evaluation

Measurements include:

- Kernel execution time
- Effective memory bandwidth
- GPU occupancy
- Arithmetic throughput
- Trit decoding overhead

## End-to-End Inference Benchmarking

Measurements include:

- Prompt processing throughput
- Token generation throughput
- End-to-end latency
- Peak memory usage

Comparisons will be made against:

- Existing TQ1_0 implementations
- FP16 execution
- Other GGML quantization formats

## Mobile Device Evaluation

Target devices:

- Apple GPUs
- Qualcomm Adreno GPUs
- ARM Mali GPUs

Measurements:

- Sustained throughput
- Thermal behavior
- GPU utilization
- Power consumption
- Memory bandwidth utilization

## Success Criteria

### Functional Requirements

- Correct BitNet/TQ1_0 execution
- Successful llama.cpp integration
- Vulkan and Metal support

### Performance Requirements

- Improved performance over generic dequantization paths
- Reduced memory bandwidth requirements
- Efficient GPU utilization
- Sustained performance under thermal constraints

### Engineering Requirements

- Maintainable GGML integration
- Maximum reuse of existing infrastructure
- Minimal impact on existing inference pathways

---
## 7. Mobile Hardware Constraints and Efficient Inference Strategies

### Memory Bandwidth Constraints

- Mobile GPUs are often bandwidth-bound rather than compute-bound.
- CPU and GPU share system memory on most smartphones.
- Frequent weight dequantization can increase memory traffic.

**Mitigation**
- Keep weights in compressed BitNet/TQ1_0 format.
- Fuse dequantization with matrix multiplication.
- Cache activations in shared/threadgroup memory.

### Memory Capacity Constraints

- Smartphones have limited RAM available for inference.
- Large models can quickly exhaust memory budgets.

**Mitigation**
- Leverage BitNet's 1.58-bit weight representation.

### Thermal Constraints

- Sustained inference workloads can trigger thermal throttling.
- Reduced clock speeds impact throughput.

**Mitigation**
- Optimize for performance-per-watt.
- Profile sustained workloads, not just peak performance.

### Battery Consumption

- Memory accesses are often more expensive than arithmetic operations.
- Long-running inference can significantly impact battery life.

**Mitigation**
- Minimize memory movement.
- Maximize on-chip data reuse.
- Reduce global memory accesses.

### Compute Constraints

- Mobile GPUs provide limited compute resources compared to desktop GPUs.
- Trit decoding overhead can become a bottleneck.

**Mitigation**
- Decode multiple trits per load.
- Use tiled matrix multiplication.
- Leverage SIMD-group/subgroup operations.

### Architecture Diversity

- Apple, Adreno, and Mali GPUs have different execution characteristics.
- A single kernel configuration may not perform optimally everywhere.

**Mitigation**
- Use backend-specific tuning.
- Optimize tile sizes and workgroup configurations per architecture.

# 8. Project Schedule, Milestones, and Deliverables

| Phase | Activities | Deliverables | Checkpoint |
|---------|------------|------------|------------|
| Investigation & Design | Study BitNet/TQ1_0 and existing kernels | Technical design document | Design review |
| PoC Development | CPU reference implementation and GPU PoC | Validation harness and prototype | Correctness validation |
| Optimization & Integration | Extend GPU kernels and integrate into llama.cpp | Integrated implementation | Performance review |
| Profiling & Benchmarking | Device profiling and tuning | Benchmark report | Target metrics achieved |
| Documentation & Delivery | Final testing and documentation | Report, source code, presentation | Final submission |

---

# 9. Risks and Mitigation Strategies

| Risk | Impact | Mitigation |
|--------|---------|---------|
| Trit decoding overhead | Reduced throughput | Fuse decoding with computation |
| GPU architectural differences | Portability challenges | Backend-specific tuning |
| Memory bandwidth bottlenecks | Reduced performance | Shared memory and caching |
| Thermal throttling | Reduced sustained throughput | Performance-per-watt optimization |
| llama.cpp integration complexity | Schedule delays | Reuse existing TQ1_0 infrastructure |

---

# 10. Assumptions and Dependencies

- BitNet model specification is available.
- Existing TQ1_0 support can be reused and extended.
- Access to Apple, Adreno, and Mali hardware is available.
- Vulkan and Metal backends are functional.
- Scope is limited to inference.
- Existing llama.cpp infrastructure remains compatible.

The success of the project depends primarily on access to representative hardware and the ability to leverage the existing GGML implementation rather than introducing entirely new execution paths.

## 11. Proof of Concept Implementation and Validation

A PoC implementation has been developed and is available in the following branch:

**Repository:** https://github.com/Ax9D/llama.cpp/tree/bitnet-poc

The implementation extends the existing GGML Metal backend to support TQ1_0 matrix-matrix multiplication using the dequantization strategy described previously. Rather than fully dequantizing a weight block, ternary weights are decoded on-the-fly within the kernel and immediately consumed during accumulation, minimizing memory bandwidth usage and temporary storage requirements.

### Building the Validation Harness

The implementation can be validated using `test-backend-ops`, which is the standard GGML backend correctness testing framework used to compare backend implementations against the CPU reference implementation.

```bash
cmake -B build
cmake --build build -j
```

Run the following command:

```bash
./bin/test-backend-ops -o MUL_MAT -p "type_a=tq1_0.*type_b=f32.*m=16.*n=16.*k=256"
```

### Notes

* The current proof-of-concept implements the matrix-matrix (`MUL_MAT`) path.
* Not all TQ1_0 tests are expected to pass at this stage.
* Smaller matrix configurations are routed through the matrix-vector path, which has not yet been implemented.
* The focus of this proof-of-concept is validating the TQ1_0 dequantization strategy and matrix-matrix execution path.

### Validation Results

The implementation was validated using `test-backend-ops` on my personal Apple M4 system.

Results:

```text
12/12 tests passed
```

All tested `MUL_MAT(type_a=tq1_0, type_b=f32)` configurations passed successfully and matched the CPU reference implementation.

This demonstrates correctness of the TQ1_0 matrix-matrix implementation and validates the proposed dequantization approach for GPU execution.
