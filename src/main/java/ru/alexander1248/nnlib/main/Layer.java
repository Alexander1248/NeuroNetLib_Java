package ru.alexander1248.nnlib.main;


import ru.alexander1248.nnlib.kernel.CUDATraining;

public class Layer {
    private Layer prevLayer;
    private double[] input;
    private Neuron[] neurons;
    private AFunction function;
    boolean firstLayer;

    private final CalculatingType type;

    private int recurrent;

    private CUDATraining train;



    public Layer(AFunction function, Layer prevLayer, int size, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = false;
        this.prevLayer = prevLayer;
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) neurons[i] = new Neuron(function, prevLayer.neurons.length, reccurent);
        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(prevLayer.neurons.length);

    }
    public Layer(AFunction function, int inputSize, int size, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = true;
        input = new double[inputSize];
        neurons = new Neuron[size];
        for (int i = 0; i < size; i++) neurons[i] = new Neuron(function, inputSize, reccurent);
        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(inputSize);
    }

    public Layer(AFunction function, boolean[][] links, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = true;
        input = new double[links[0].length];
        neurons = new Neuron[links.length];
        for (int i = 0; i < links.length; i++) {
            int[] l = new int[links[i].length];
            for (int j = 0; j < links[i].length; j++) l[j] = links[i][j] ? 1 : 0;
            neurons[i] = new Neuron(function, l, reccurent);
        }
        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(links[0].length);
    }
    public Layer(AFunction function, Layer prevLayer, boolean[][] links, boolean reccurent, CalculatingType type) {
        this.type = type;
        firstLayer = false;
        this.prevLayer = prevLayer;
        neurons = new Neuron[links.length];
        for (int i = 0; i < links.length; i++) {
            int[] l = new int[links[i].length];
            for (int j = 0; j < links[i].length; j++) l[j] = links[i][j] ? 1 : 0;
            neurons[i] = new Neuron(function, l, reccurent);
        }
        this.function = function;
        if (type.equals(CalculatingType.GPU)) train = new CUDATraining(prevLayer.neurons.length);
    }


    private void CL_CPU() {
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
    private void CL_GPU() {
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

    public void calculateLayer() {
        switch (type) {
            case CPU -> CL_CPU();
            case GPU -> CL_GPU();
        }
    }

    public void calculateOutLayerError(double[] rightResults) {
        for (int i = 0; i < neurons.length; i++)
            neurons[i].setError((rightResults[i] - neurons[i].getOutput()) * ActivationFunction.GetDerivative(neurons[i].getFunction(), neurons[i].getWeightedSum()));
    }
    public void calculateInOrHiddenLayerError(Layer postLayer) {
        for (int i = 0; i < neurons.length; i++) {
            double error = 0;
            for (int j = 0; j < postLayer.neurons.length; j++) {
                error += postLayer.neurons[j].weights[i] * postLayer.neurons[j].getError();
                error += recurrent * postLayer.neurons[j].recurrent * postLayer.neurons[j].getError();
            }
            neurons[i].setError(error * ActivationFunction.GetDerivative(neurons[i].getFunction(),neurons[i].getWeightedSum()));
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
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < input.length; j++) {
                    neurons[i].acceleration[j] *= momentumCoefficient;
                    neurons[i].acceleration[j] += neurons[i].getLinks()[j] * (1 - momentumCoefficient) * neurons[i].getError() * input[j] * trainSpeed;
                    neurons[i].weights[j] += neurons[i].getLinks()[j] * neurons[i].acceleration[j];
                }
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
                neurons[i].recurrent += recurrent * neurons[i].getError() * neurons[i].getOutput() * trainSpeed;
            }
        }
        else {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < prevLayer.neurons.length; j++) {
                    neurons[i].acceleration[j] *= momentumCoefficient;
                    neurons[i].acceleration[j] += (1 - momentumCoefficient) * neurons[i].getError() * prevLayer.neurons[j].getOutput() * trainSpeed;
                    neurons[i].weights[j] += neurons[i].getLinks()[j] * neurons[i].acceleration[j];
                }
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
                neurons[i].recurrent += recurrent * neurons[i].getError() * neurons[i].getOutput() * trainSpeed;
            }
        }
    }
    public void CNW_GPU(double trainSpeed, double momentumCoefficient) {
        if (firstLayer) {
            for (int i = 0; i < neurons.length; i++) {
                train.run(input, neurons[i].acceleration, neurons[i].weights, neurons[i].getLinks(), neurons[i].getError(), trainSpeed, momentumCoefficient);
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
                neurons[i].recurrent += recurrent * neurons[i].getError() * neurons[i].getOutput() * trainSpeed;
            }
        }
        else {
            for (int i = 0; i < neurons.length; i++) {
                for (int j = 0; j < prevLayer.neurons.length; j++) {
                    neurons[i].acceleration[j] *= momentumCoefficient;
                    neurons[i].acceleration[j] += (1 - momentumCoefficient) * neurons[i].getError() * prevLayer.neurons[j].getOutput() * trainSpeed;
                    neurons[i].weights[j] += neurons[i].getLinks()[j] * neurons[i].acceleration[j];
                }
                neurons[i].biasWeight += neurons[i].getError() * trainSpeed;
                neurons[i].recurrent += recurrent * neurons[i].getError() * neurons[i].getOutput() * trainSpeed;
            }
        }
    }


    public void mutate(double coefficient) {
        for (int i = 0; i < 10; i++)
            neurons[(int) (Math.random() * neurons.length)].mutate(coefficient / 10);
    }



    public int getInputSize() {
        return input.length;
    }
    public double getOutput(int i) {
        if (i >= 0 && i < this.neurons.length) return neurons[i].getOutput();
        return Double.MIN_EXPONENT;
    }
    public Neuron[] getNeurons() {
        return neurons;
    }
    public AFunction getFunction() {
        return function;
    }
    public boolean getRecurrency() {
        return recurrent == 1;
    }
    public Layer getPrevLayer() {
        if(!firstLayer) return prevLayer;
        else return null;
    }

    public void setInput(int i, double input) {
        if (firstLayer && i >= 0 && i < this.input.length) this.input[i] = input;
    }
}
