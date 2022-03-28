package ru.nnlib.core.network;

import ru.nnlib.core.functions.ActivationFunction;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Neuron {
    private final ActivationFunction function;

    private final List<Double> inputs;
    private final List<Double> weights;

    private double output;

    public Neuron(ActivationFunction function) {
        this.function = function;
        inputs = new LinkedList<>();
        weights = new LinkedList<>();
    }


    public ActivationFunction getFunction() {
        return function;
    }

    public double getOutput() {
        return output;
    }

    public List<Double> getWeights() {
        return weights;
    }

    public void setInput(int i, double val) {
        inputs.set(i, val);
    }

    public void editWeight(int i, double delta) {
        weights.set(i, weights.get(i) + delta);
    }

    public void addInput() {
        inputs.add(0.0);
        weights.add(Math.random() - 0.5);
    }


}
