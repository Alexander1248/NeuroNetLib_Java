extern "C"

 __global__ void train(float *data, float *acceleration, float *weights, int *links, float error, float momentumCoefficient, float trainSpeed)
{
    int i = threadIdx.x;
    acceleration[i] *= momentumCoefficient;
    acceleration[i] += links[i] * (1 - momentumCoefficient) * error * data[i] * trainSpeed;
    weights[i] += links[i] * acceleration[i];
}