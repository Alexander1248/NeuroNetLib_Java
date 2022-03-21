package ru.alexander1248.nnlib.main;

public class Neuron {
    private AFunction function;
    protected double[] input;

    protected double[] weights;
    protected double biasWeight;
    protected double[] acceleration;

    private double weightedSum;
    private double output;
    private double error;

    private final int[] links;

    protected double recurrent;
    private boolean rec;


    public Neuron(AFunction function, int size, boolean reccurent) {
        this.function = function;
        input = new double[size];
        weights = new double[size];
        acceleration = new double[size];
        this.links = new int[size];
        for (int i = 0; i < input.length; i++) {
            links[i] = 1;
            weights[i] = Math.random() * 2 - 1;
        }
        biasWeight = Math.random() * 2 - 1;
        this.recurrent = reccurent ? Math.random() * 2 - 1 : 0;
        rec = reccurent;
    }
    public Neuron(AFunction function, int[] links, boolean reccurent) {
        this.function = function;
        input = new double[links.length];
        weights = new double[links.length];
        acceleration = new double[links.length];
        this.links = links.clone();
        for (int i = 0; i < input.length; i++) weights[i] = Math.random() * 2 - 1;
        biasWeight = Math.random() * 2 - 1;
        this.recurrent = reccurent ? Math.random() * 2 - 1 : 0;
        rec = reccurent;
    }
    //Getters
    public double[] getInput() {
        return input;
    }
    public int getInputSize() {
        return input.length;
    }

    public double getOutput() {
        return output;
    }

    public double getWeightedSum() { return weightedSum; }

    public AFunction getFunction() {
        return function;
    }

    public double getError() { return error; }

    public double[] getWeights() {
        return weights;
    }

    public double getBiasWeight() {
        return biasWeight;
    }

    //Setters
    public void setWeight(int i, double weight) {
        this.weights[i] = weight;
    }

    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    public void setInput(int i, double input) {
        this.input[i] = input;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    void setFunction(AFunction function) {
        this.function = function;
    }

    public void setError(double error) { this.error = error; }

    public int[] getLinks() {
        return links;
    }

    //Modifiers
    public void modifyWeights(int i, double weightDelta) {
        weights[i] += weightDelta;
    }

    //========================================
    //                 Math
    //========================================
    public void calculateNeuron() {
        weightedSum = output * recurrent;
        for(int i = 0; i < input.length; i++) weightedSum += input[i] * links[i] * weights[i];
        weightedSum += biasWeight;
        output = ActivationFunction.GetFunction(function, weightedSum);
    }

    public void mutate(double coefficient) {
        if (Math.random() < coefficient) biasWeight = Math.random() * 2 - 1;
        if (rec && Math.random() < coefficient) recurrent = Math.random() * 2 - 1;
        for (int i = 0; i < weights.length; i++)
            if (Math.random() < coefficient) weights[i] = Math.random() * 2 - 1;

    }
}
