extern "C"


__global__ void calculate(double weightedSum, double *input, double *weights, int *links)
{
    int i = threadIdx.x;
    weightedSum += input[i] * links[i] * weights[i];
}