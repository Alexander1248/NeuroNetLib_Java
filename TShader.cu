extern "C"
 __global__ void train(int len, double *data, double *acceleration, double *weights, int *links, double error, double momentumCoefficient, double trainSpeed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        acceleration[i] *= momentumCoefficient;
        acceleration[i] += links[i] * (1 - momentumCoefficient) * error * data[i] * trainSpeed;
        weights[i] += links[i] * acceleration[i];
    }
}