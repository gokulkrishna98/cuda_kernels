#!POPCORN leaderboard sort 

# This is a submission template for popcorn leaderboard 'sort'.
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
__global__
void forward_kernel(float* A, int len, int j) {
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int l = t ^ j;
  int dir = t & len;

  if(t < l){
    if(((dir == 0) && A[t] > A[l]) || ((dir !=0) && A[t] < A[l])){
        float temp = A[t];
        A[t] = A[l];
        A[l] = temp;
    }
  }

  return;
}

torch::Tensor forward(torch::Tensor A) {
    int n = static_cast<int>(A.size(0));

    torch::Tensor O = A.clone();

    int num_threads = 1024;
    int num_blocks = n/1024;
    dim3 grid_dim(num_blocks);
    dim3 block_dim(num_threads);

    for(int len=2; len <= n; len *= 2){
        for(int j=len/2; j>0; j /= 2){
            forward_kernel<<<grid_dim, block_dim>>>(
                O.data_ptr<float>(),
                len,
                j
            );
            cudaDeviceSynchronize();
        }
    }

    return O;
}
'''

cpp_src = 'torch::Tensor forward(torch::Tensor A);'
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"

bit_sort = load_inline(
    name='bit_sort',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['forward'],
    with_cuda=True,
    extra_cuda_cflags=["-arch=sm_75"]
)


def custom_kernel(data: input_t) -> output_t:
    a = data
    return bit_sort.forward(a)