extern "C"
 __global__ void train(int len, int prevLen, double *data, double *acceleration, double *weights, int *links, double *error, double momentumCoefficient, double trainSpeed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len * prevLen) {
//         acceleration[i] *= momentumCoefficient;
//         acceleration[i] += links[i] * (1 - momentumCoefficient) * error[y] * data[x] * trainSpeed;
//         weights[i] += links[i] * acceleration[i];
    }
}