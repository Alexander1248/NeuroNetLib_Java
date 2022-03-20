package ru.alexander1248.nnlib;

import java.util.Arrays;
import java.util.Objects;

public class Neuron {
    private AFunction function;
    protected double[] input;

    protected double[] weights;
    protected double biasWeight;
    protected double[] acceleration;

    private double weightedSum;
    private double output;
    private double error;


    public Neuron(AFunction function, int size) {
        this.function = function;
        input = new double[size];
        weights = new double[size];
        acceleration = new double[size];
        for (int i = 0; i < input.length; i++) weights[i] = Math.random() * 2 - 1;
        biasWeight = Math.random() * 2 - 1;
    }
    public Neuron(AFunction function, boolean[] links) {
        this.function = function;
        input = new double[links.length];
        weights = new double[links.length];
        acceleration = new double[links.length];
        for (int i = 0; i < input.length; i++) weights[i] = Math.random() * 2 - 1;
        biasWeight = Math.random() * 2 - 1;
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

    //Modifiers
    public void modifyWeights(int i, double weightDelta) {
        weights[i] += weightDelta;
    }

    //========================================
    //                 Math
    //========================================
    public void calculateNeuron() {
        weightedSum = 0;
        for(int i = 0; i < input.length; i++) weightedSum += input[i] * weights[i];
        weightedSum += biasWeight;
        output = ActivationFunction.GetFunction(function, weightedSum);
    }

    public void mutate(double coefficient) {
        if (Math.random() < coefficient) biasWeight = Math.random() * 2 - 1;
        for (int i = 0; i < weights.length; i++)
            if (Math.random() < coefficient) weights[i] = Math.random() * 2 - 1;
    }
}
