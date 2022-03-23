package ru.nnlib.core;


import ru.nnlib.kernel.CUDATraining;
import ru.nnlib.kernel.CalculatingType;

public class Layer {
    private Layer prevLayer;
    private double[] input;
    private final AFunction function;
    boolean firstLayer;

    private final CalculatingType type;

    private final double[] weights;
    private final double[] biasWeight;
    private final double[] acceleration;

    private final double[] weightedSum;
    private final double[] output;
    private final double[] error;

    private final int[] links;
    boolean[][] l;

    private final double[] recurrent;
    private final int rec;

    CUDATraining train;

    public Layer(AFunction function, Layer prevLayer, int size, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = false;
        this.prevLayer = prevLayer;

        biasWeight = new double[size];
        weights = new double[size * prevLayer.output.length];
        acceleration = new double[size * prevLayer.output.length];
        links = new int[size * prevLayer.output.length];
        this.recurrent = new double[size];
        weightedSum = new double[size];
        output = new double[size];
        error = new double[size];
        for (int i = 0; i < size; i++) {
            biasWeight[i] = Math.random() * 2 - 1;
            this.recurrent[i] = Math.random() * 2 - 1;
            for (int j = 0; j < prevLayer.output.length; j++) {
                weights[i * prevLayer.output.length + j] = Math.random() * 2 - 1;
                links[i * prevLayer.output.length + j] = 1;
            }
        }
        rec = reccurent ? 1 : 0;

        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(error.length, prevLayer.output.length);

    }
    public Layer(AFunction function, int inputSize, int size, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = true;
        input = new double[inputSize];

        biasWeight = new double[size];
        weights = new double[size * inputSize];
        acceleration = new double[size * inputSize];
        links = new int[size * inputSize];
        this.recurrent = new double[size];
        weightedSum = new double[size];
        output = new double[size];
        error = new double[size];
        for (int i = 0; i < size; i++) {
            biasWeight[i] = Math.random() * 2 - 1;
            this.recurrent[i] = Math.random() * 2 - 1;
            for (int j = 0; j < inputSize; j++) {
                weights[i * inputSize + j] = Math.random() * 2 - 1;
                links[i * inputSize + j] = 1;
            }
        }
        rec = reccurent ? 1 : 0;

        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(error.length, input.length);
    }

    public Layer(AFunction function, boolean[][] links, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = true;
        input = new double[links[0].length];

        biasWeight = new double[links.length];
        weights = new double[links.length * links[0].length];
        acceleration = new double[links.length * links[0].length];
        this.links = new int[links.length * links[0].length];
        this.recurrent = new double[links.length];
        weightedSum = new double[links.length];
        output = new double[links.length];
        error = new double[links.length];
        for (int i = 0; i < links.length; i++) {
            biasWeight[i] = Math.random() * 2 - 1;
            this.recurrent[i] = Math.random() * 2 - 1;
            for (int j = 0; j < links[0].length; j++) {
                weights[j * links[0].length + i] = Math.random() * 2 - 1;
                this.links[i * links[0].length + j] = links[i][j] ? 1 : 0;
            }
        }
        rec = reccurent ? 1 : 0;
        l = links.clone();

        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(error.length, prevLayer.output.length);
    }
    public Layer(AFunction function, Layer prevLayer, boolean[][] links, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = false;
        this.prevLayer = prevLayer;

        biasWeight = new double[links.length];
        weights = new double[links.length * links[0].length];
        acceleration = new double[links.length * links[0].length];
        this.links = new int[links.length * links[0].length];
        this.recurrent = new double[links.length];
        weightedSum = new double[links.length];
        output = new double[links.length];
        error = new double[links.length];
        for (int i = 0; i < links.length; i++) {
            biasWeight[i] = Math.random() * 2 - 1;
            this.recurrent[i] = Math.random() * 2 - 1;
            for (int j = 0; j < links[0].length; j++) {
                weights[j * links[0].length + i] = Math.random() * 2 - 1;
                this.links[i * links[0].length + j] = links[i][j] ? 1 : 0;
            }
        }
        rec = reccurent ? 1 : 0;
        l = links.clone();

        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(error.length, input.length);
    }


    private void CL_CPU() {
        if (firstLayer) {
            for (int i = 0; i < error.length; i++) {
                weightedSum[i] = output[i] * recurrent[i] + biasWeight[i];
                for(int j = 0; j < input.length; j++) weightedSum[i] += input[j] * links[i * input.length + j] * weights[i * input.length + j];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        } else {
            for (int i = 0; i < error.length; i++) {
                weightedSum[i] = output[i] * recurrent[i] + biasWeight[i];
                for(int j = 0; j < prevLayer.getLength(); j++) weightedSum[i] += prevLayer.output[j] * links[i * prevLayer.getLength() + j] * weights[i * prevLayer.getLength() + j];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        }
    }
    private void CL_GPU() {
        if (firstLayer) {
            for (int i = 0; i < error.length; i++) {
                weightedSum[i] = output[i] * recurrent[i] + biasWeight[i];
                for(int j = 0; j < input.length; j++) weightedSum[i] += input[j] * links[i * input.length + j] * weights[i * input.length + j];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        } else {
            for (int i = 0; i < error.length; i++) {
                weightedSum[i] = output[i] * recurrent[i] + biasWeight[i];
                for(int j = 0; j < prevLayer.output.length; j++) weightedSum[i] += prevLayer.output[j] * links[i * prevLayer.error.length + j] * weights[i * prevLayer.error.length + j];
                output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
            }
        }
        for (int i = 0; i < error.length; i++) output[i] = ActivationFunction.GetFunction(function, weightedSum[i]);
    }

    public void calculateLayer() {
        switch (type) {
            case CPU -> CL_CPU();
            case GPU -> CL_GPU();
        }
    }

    public void calculateOutLayerError(double[] rightResults) {
        for (int i = 0; i < error.length; i++)
            error[i] = (rightResults[i] - output[i]) * ActivationFunction.GetDerivative(function, weightedSum[i]);
    }
    public void calculateInOrHiddenLayerError(Layer postLayer) {
        for (int i = 0; i < error.length; i++) {
            double error = 0;
            for (int j = 0; j < postLayer.error.length; j++) {
                error += postLayer.weights[j * this.error.length + i] * postLayer.error[j];
                error += rec * postLayer.recurrent[j] * postLayer.error[j];
            }
            this.error[i] = error * ActivationFunction.GetDerivative(function, weightedSum[i]);
        }
    }

    public void calculateNewWeights(double trainSpeed, double momentumCoefficient) {
        switch (type) {
            case CPU -> CNW_CPU(trainSpeed, momentumCoefficient);
            case GPU -> CNW_GPU(trainSpeed, momentumCoefficient);
        }
    }

    public void CNW_CPU(double trainSpeed, double momentumCoefficient) {
        if (firstLayer) {
            for (int i = 0; i < error.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    acceleration[i * input.length + j] *= momentumCoefficient;
                    acceleration[i * input.length + j] += links[i * input.length + j] * (1 - momentumCoefficient) * error[i] * input[j] * trainSpeed;
                    weights[i * input.length + j] += links[i * input.length + j] * acceleration[i * input.length + j];
                }
                biasWeight[i] += error[i] * trainSpeed;
                recurrent[i] += rec * error[i] * output[i] * trainSpeed;
            }
        }
        else {
            for (int i = 0; i < error.length; i++) {
                for (int j = 0; j < prevLayer.error.length; j++) {
                    acceleration[i * prevLayer.error.length + j] *= momentumCoefficient;
                    acceleration[i * prevLayer.error.length + j] += links[i * prevLayer.error.length + j] * (1 - momentumCoefficient) * error[i] * prevLayer.output[j] * trainSpeed;
                    weights[i * prevLayer.error.length + j] += links[i * prevLayer.error.length + j] * acceleration[i * prevLayer.error.length + j];
                }
                biasWeight[i] += error[i] * trainSpeed;
                recurrent[i] += rec * error[i] * output[i] * trainSpeed;
            }
        }
    }
    public void CNW_GPU(double trainSpeed, double momentumCoefficient) {
        if (firstLayer) {
            train.run(input, acceleration, weights, links, error, trainSpeed, momentumCoefficient);
            for (int i = 0; i < output.length; i++) {
                biasWeight[i] += error[i] * trainSpeed;
                recurrent[i] += rec * error[i] * output[i] * trainSpeed;
            }
        }
        else {
            train.run(prevLayer.output, acceleration, weights, links, error, trainSpeed, momentumCoefficient);

            for (int i = 0; i < output.length; i++) {
                biasWeight[i] += error[i] * trainSpeed;
                recurrent[i] += rec * error[i] * output[i] * trainSpeed;
            }
        }
    }


    public void mutate(double coefficient) {
        for (int i = 0; i < 10; i++) {
            if (Math.random() < coefficient) biasWeight[i] = Math.random() * 2 - 1;
            if (rec == 1 && Math.random() < coefficient) recurrent[i] = Math.random() * 2 - 1;
            for (int j = 0; j < weights.length; j++)
                if (Math.random() < coefficient) weights[i * prevLayer.error.length + j] = Math.random() * 2 - 1;
        }

    }



    public int getInputSize() {
        return input.length;
    }
    public double getOutput(int i) {
        if (i >= 0 && i < output.length) return output[i];
        return Double.MIN_EXPONENT;
    }
    public AFunction getFunction() {
        return function;
    }
    public boolean getRecurrent() {
        return rec == 1;
    }

    public void setInput(int i, double input) {
        if (firstLayer && i >= 0 && i < this.input.length) this.input[i] = input;
    }

    public int getLength() {
        return error.length;
    }

    public double[] getWeights() {
        return weights;
    }

    public double[] getBiasWeight() {
        return biasWeight;
    }
}
