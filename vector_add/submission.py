#!POPCORN leaderboard vectoradd

# This is a submission template for popcorn leaderboard 'vectoradd'.
# Your task is as follows:
# > Implement a float16 vector addition kernel.

# > 

# > Input: tuple(torch.Tensor, torch.Tensor) with tensors of shape (N, N) and type torch.float16. These tensors are from

# > a normal distribution with mean 0 and variance 1.

# > Output: torch.Tensor of shape (N, N) and type torch.float16

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
from typing import List

import os
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = '''
#include <cuda_fp16.h>
#include <torch/extension.h>
__global__
void forward_kernel(half* A, half* B, half* O, int n, int m) {
    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.x;

    if (x < m) {
        O[x*m + y] = __hadd(A[x*m + y], B[x*m + y]);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    auto O = torch::zeros_like(A);
    torch::Device device(torch::kCUDA);
    O = O.to(device);

    int n = static_cast<int>(A.size(0));
    int m = static_cast<int>(A.size(1));

    int num_threads = 1024;
    dim3 grid_dim(m, (num_threads + n - 1)/num_threads);
    dim3 block_dim(num_threads);

    forward_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<half*>(A.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(B.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(O.data_ptr<c10::Half>()),
        n,
        m
    );

    return O;
}
'''

cpp_src = 'torch::Tensor forward(torch::Tensor A, torch::Tensor B);'
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"

vec_add = load_inline(
    name='vec_add',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['forward'],
    with_cuda=True,
    extra_cuda_cflags=["-arch=sm_75"]
)


def custom_kernel(data: input_t) -> output_t:
    a, b = data
    assert a.is_cuda and a.is_cuda, "Input tensors must be on GPU"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "Input tensors must be float16"
    return vec_add.forward(a, b)