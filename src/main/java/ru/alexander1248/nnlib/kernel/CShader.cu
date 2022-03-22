extern "C"

__global__ void calculate(float weightedSum, float *input, int *links, float *weights)
{
    int i = threadIdx.x;
    weightedSum += input[i] * links[i] * weights[i];
}