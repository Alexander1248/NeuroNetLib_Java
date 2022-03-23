extern "C"
 __global__ void train(int len, int prevLen, double *data, double *acceleration, double *weights, int *links, double *error, double momentumCoefficient, double trainSpeed)
{
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int i = x * prevLen + y;
//     if (x < prevLen && y < len)
//     acceleration[i] *= momentumCoefficient;
//     acceleration[i] += links[i] * (1 - momentumCoefficient) * error[y] * data[x] * trainSpeed;
//     weights[i] += links[i] * acceleration[i];
}