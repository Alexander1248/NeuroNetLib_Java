package ru.alexander1248.nnlib;


import com.aparapi.Kernel;
import com.aparapi.device.Device;
import com.aparapi.exception.CompileFailedException;

public class Layer {
    private final Layer prevLayer;
    private final double[] input;


    protected double[][] acceleration;
    protected double[][] weights;
    protected double[] biasWeight;

    private final double[] weightedSum;
    private final double[] output;
    private final double[] error;

    private final AFunction function;
    private final boolean firstLayer;

    private Kernel[] kernels;
    private Thread[] threads;

    private CalculatingType type = CalculatingType.CPU1;

    public Layer(Layer prevLayer, AFunction function, int size) {
        firstLayer = false;
        this.prevLayer = prevLayer;
        this.function = function;

        acceleration = new double[size][prevLayer.output.length];
        weights = new double[size][prevLayer.output.length];
        biasWeight = new double[size];

        weightedSum = new double[size];
        output = new double[size];
        error = new double[size];
        input = new double[0];

        kernels = new Kernel[size];
        threads = new Thread[size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < prevLayer.output.length; j++) weights[i][j] = Math.random() * 2 - 1;
            biasWeight[i] = Math.random() * 2 - 1;
        }

    }
    public Layer(int inputSize, AFunction function, int size) {
        firstLayer = true;
        prevLayer = null;
        this.function = function;

        acceleration = new double[size][inputSize];
        weights = new double[size][inputSize];
        biasWeight = new double[size];

        weightedSum = new double[size];
        output = new double[size];
        error = new double[size];
        input = new double[inputSize];

        kernels = new Kernel[size];
        threads = new Thread[size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < input.length; j++) weights[i][j] = Math.random() * 2 - 1;
            biasWeight[i] = Math.random() * 2 - 1;
        }
    }

    public void setInput(int i, double input) {
        if (firstLayer && i >= 0 && i < this.input.length) this.input[i] = input;
    }
    public double getOutput(int i) {
        if (i >= 0 && i < output.length) return output[i];
        return Double.MIN_EXPONENT;
    }
    public Layer getPrevLayer() {
       return prevLayer;
    }


    public void precompileCL() {
        kernels = new Kernel[output.length];
        switch (type) {
            case CPU1, CPU2, CPU4, CPU8, CPU16 -> {
                threads = new Thread[output.length];
                for (int i = 0; i < output.length; i++)
                    compileCPU_CL(i);
            }
            case GPU -> {
                kernels = new Kernel[output.length];
                for (int i = 0; i < output.length; i++)
                    compileGPU_CL(i);
            }
        }
    }

    private void compileCPU_CL(int i) {
        threads[i] = new Thread(() -> {
            weightedSum[i] = 0;
            for (int j = 0; j < input.length; j++) weightedSum[i] += input[j] * weights[i][j];
            weightedSum[i] += biasWeight[i];
            output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
        });
    }
    private void compileGPU_CL(int i) {
        try {
            kernels[i] = new Kernel() {
                @Override
                public void run() {
                    weightedSum[i] = 0;
                    for (int j = 0; j < input.length; j++) weightedSum[i] += input[j] * weights[i][j];
                    weightedSum[i] += biasWeight[i];
                    output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
                }
            }.compile(Device.bestGPU());
        } catch (CompileFailedException ex) {
            ex.printStackTrace();
        }
    }


    private void compileCPU_CNW(int i, double trainSpeed, double momentumCoefficient) {
        if (firstLayer) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < input.length; j++) {
                    acceleration[i][j] *= momentumCoefficient;
                    acceleration[i][j] += (1 - momentumCoefficient) * error[i] * input[j] * trainSpeed;
                    weights[i][j] += acceleration[i][j];
                }
                biasWeight[i] += error[i] * trainSpeed;
            });
            threads[i].setPriority(Thread.MAX_PRIORITY);
        }
        else {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < input.length; j++) {
                    acceleration[i][j] *= momentumCoefficient;
                    acceleration[i][j] += (1 - momentumCoefficient) * error[i] * prevLayer.output[j] * trainSpeed;
                    weights[i][j] += acceleration[i][j];
                }
                biasWeight[i] += error[i] * trainSpeed;
            });
            threads[i].setPriority(Thread.MAX_PRIORITY);
        }
    }

    public void calculateLayer() {
        CLMono();
    }
    public void CLMono() {
        if (firstLayer) {
            for (int i = 0; i < output.length; i++) {
                weightedSum[i] = 0;
                for (int j = 0; j < input.length; j++) weightedSum[i] += input[j] * weights[i][j];
                weightedSum[i] += biasWeight[i];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        } else {
            for (int i = 0; i < output.length; i++) {
                weightedSum[i] = 0;
                for (int j = 0; j <  prevLayer.output.length; j++) weightedSum[i] += prevLayer.output[j] * weights[i][j];
                weightedSum[i] += biasWeight[i];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        }
    }

    public void calculateOutLayerError(double[] rightResults) {
        for (int i = 0; i < error.length; i++)
            error[i] = (rightResults[i] - output[i]) * ActivationFunction.GetDerivative(function, weightedSum[i]);
    }
    public void calculateInOrHiddenLayerError(Layer postLayer) {
        for (int i = 0; i < output.length; i++) {
            double err = 0;
            for (int j = 0; j < postLayer.output.length; j++) {
                err += postLayer.weights[j][i] * postLayer.error[j];
            }
            error[i] = err * ActivationFunction.GetDerivative(function, weightedSum[i]);
        }
    }
    public void calculateNewWeights(double trainSpeed, double momentumCoefficient) {
        switch (type) {
            case CPU1 -> CNWMono(trainSpeed, momentumCoefficient);
            case CPU2 -> CNWMulti(trainSpeed, momentumCoefficient, 2);
            case CPU4 -> CNWMulti(trainSpeed, momentumCoefficient, 4);
            case CPU8 -> CNWMulti(trainSpeed, momentumCoefficient, 8);
            case CPU16 -> CNWMulti(trainSpeed, momentumCoefficient, 16);
            case GPU -> CNWGPU(trainSpeed, momentumCoefficient);
        }
    }

    private void CNWMono(double trainSpeed, double momentumCoefficient) {
        if (firstLayer) {
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    acceleration[i][j] *= momentumCoefficient;
                    acceleration[i][j] += (1 - momentumCoefficient) * error[i] * input[j] * trainSpeed;
                    weights[i][j] += acceleration[i][j];
                }
                biasWeight[i] += error[i] * trainSpeed;
            }
        }
        else {
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < prevLayer.output.length; j++) {
                    acceleration[i][j] *= momentumCoefficient;
                    acceleration[i][j] += (1 - momentumCoefficient) * error[i] * prevLayer.output[j] * trainSpeed;
                    weights[i][j] += acceleration[i][j];
                }
                biasWeight[i] += error[i] * trainSpeed;
            }
        }
    }
    private void CNWMulti(double trainSpeed, double momentumCoefficient, int numThreads) {
        Thread[] threads = new Thread[numThreads];
        if (firstLayer) {
            int t = 0;
            for (int i = 0; i < output.length; i++) {
                if (threads[i] == null) compileCPU_CNW(i, trainSpeed, momentumCoefficient);
                while (threads[i].isAlive()) {}
                t++;
                threads[i].start();
            }
        }
        else {
            for (int i = 0; i < output.length; i++) {
                int ti = i % numThreads;
                while (threads[ti] != null && threads[ti].isAlive()) {}
                int finalI = i;
                threads[ti] = new Thread(() -> {
                    for (int j = 0; j < prevLayer.output.length; j++) {
                        acceleration[finalI][j] *= momentumCoefficient;
                        acceleration[finalI][j] += (1 - momentumCoefficient) * error[finalI] * prevLayer.output[j] * trainSpeed;
                        weights[finalI][j] += acceleration[finalI][j];
                    }
                    biasWeight[finalI] += error[finalI] * trainSpeed;
                });
                threads[ti].setPriority(Thread.MAX_PRIORITY);
                threads[ti].start();
            }
        }
        while (threads[(output.length - 1) % numThreads].isAlive()) {}
    }
    private void CNWGPU(double trainSpeed, double momentumCoefficient) {

        if (firstLayer) {
            double[] in = input;
            for (int i = 0; i < output.length; i++) {
                double[] a = acceleration[i];
                double[] w = weights[i];
                double e = error[i];
                if (kernels[i] == null) {
                    try {
                        kernels[i] = new Kernel() {
                            @Override
                            public void run() {
                                int id = getGlobalId();
                                a[id] *= momentumCoefficient;
                                a[id] += (1 - momentumCoefficient) * e * in[id] * trainSpeed;
                                w[id] += a[id];
                            }
                        }.compile(Device.bestGPU());
                    } catch (CompileFailedException ex) {
                        ex.printStackTrace();
                    }
                }
                kernels[i].setExplicit(true);
                kernels[i].put(a);
                kernels[i].put(w);
                kernels[i].put(in);

                kernels[i].execute(prevLayer.output.length);

                kernels[i].get(a);
                kernels[i].get(w);
            }
        }
        else {
            double[] in = prevLayer.output;
            for (int i = 0; i < output.length; i++) {
                double[] a = acceleration[i];
                double[] w = weights[i];
                double e = error[i];
                if (kernels[i] == null) {
                    try {
                        kernels[i] = new Kernel() {
                            @Override
                            public void run() {
                                int id = getGlobalId();
                                a[id] *= momentumCoefficient;
                                a[id] += (1 - momentumCoefficient) * e * in[id] * trainSpeed;
                                w[id] += a[id];
                            }
                        }.compile(Device.bestGPU());
                    } catch (CompileFailedException ex) {
                        ex.printStackTrace();
                    }
                }
                kernels[i].setExplicit(true);
                kernels[i].put(a);
                kernels[i].put(w);
                kernels[i].put(in);

                kernels[i].execute(prevLayer.output.length);

                kernels[i].get(a);
                kernels[i].get(w);
                acceleration[i] = a;
                weights[i] = w;
            }
        }
    }

    public int getInputSize() {
        return input.length;
    }


    public AFunction getFunction() {
        return function;
    }

    public void setCalculatingType(CalculatingType type) {
        this.type = type;
    }

    public int getSize() {
        return error.length;
    }
}
