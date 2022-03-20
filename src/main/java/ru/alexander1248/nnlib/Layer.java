package ru.alexander1248.nnlib;

import ru.alexander1248.nnlib.Neuron;

public class Layer {
    private Layer prevLayer;
    private double[] input;
    private Neuron[] neurons;
    private AFunction function;
    boolean firstLayer;

    private int numThreads = 1;

    public Layer(Layer prevLayer, AFunction function, int size) {
        firstLayer = false;
        this.prevLayer = prevLayer;
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) neurons[i] = new Neuron(function, prevLayer.neurons.length);
        this.function = function;
    }
    public Layer(int InputSize, AFunction function, int size) {
        firstLayer = true;
        input = new double[InputSize];
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) neurons[i] = new Neuron(function, InputSize);
        this.function = function;
    }

    public void setInput(int i, double input) {
        if (firstLayer && i >= 0 && i < this.input.length) this.input[i] = input;
    }
    public double getOutput(int i) {
        if (i >= 0 && i < this.neurons.length) return neurons[i].getOutput();
        return Double.MIN_EXPONENT;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public Layer getPrevLayer() {
       if(!firstLayer) return prevLayer;
       else return null;
    }


    public void calculateLayer() {
        Thread[] threads = new Thread[numThreads];
        if (firstLayer) {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < input.length; j++)
                    neurons[i].setInput(j, input[j]);
                neurons[i].calculateNeuron();
            }
        } else {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < prevLayer.neurons.length; j++)
                    neurons[i].setInput(j, prevLayer.neurons[j].getOutput());
                neurons[i].calculateNeuron();
            }
        }
    }

    public void calculateOutLayerError(double[] rightResults) {
        for (int i = 0; i < neurons.length; i++) neurons[i].setError((rightResults[i] - neurons[i].getOutput()) * ActivationFunction.GetDerivative(neurons[i].getFunction(),neurons[i].getWeightedSum()));
    }
    public void calculateInOrHiddenLayerError(Layer postLayer) {
        for (int i = 0; i < neurons.length; i++) {
            double error = 0;
            for (int j = 0; j < postLayer.neurons.length; j++) {
                error += postLayer.neurons[j].weights[i] * postLayer.neurons[j].getError();
            }
            neurons[i].setError(error * ActivationFunction.GetDerivative(neurons[i].getFunction(),neurons[i].getWeightedSum()));
        }
    }

    public void calculateNewWeights(double trainSpeed, double momentumCoef) {
        Thread[] threads = new Thread[numThreads];
        if (firstLayer) {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < prevLayer.neurons.length; j++) {
                    neurons[i].acceleration[j] *= momentumCoef;
                    neurons[i].acceleration[j] += (1 - momentumCoef) * neurons[i].getError() * input[j] * trainSpeed;
                    neurons[i].weights[j] += neurons[i].acceleration[j];
                }
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
            }
        }
        else {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < prevLayer.neurons.length; j++) {
                    neurons[i].acceleration[j] *= momentumCoef;
                    neurons[i].acceleration[j] += (1 - momentumCoef) * neurons[i].getError() * prevLayer.neurons[j].getOutput() * trainSpeed;
                    neurons[i].weights[j] += neurons[i].acceleration[j];
                }
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
            }
        }
    }

    public int getInputSize() {
        return input.length;
    }


    public AFunction getFunction() {
        return function;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }
}
