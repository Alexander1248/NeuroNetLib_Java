#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
__global__ void calculate(int len, double weightedSum, double *input, double *weights, int *links)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) weightedSum += input[i] * links[i] * weights[i];
}