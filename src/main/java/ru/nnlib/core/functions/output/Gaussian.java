package ru.nnlib.core.functions.output;

public class Gaussian extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return Math.exp(-steepness * input * input);
    }

    @Override
    public double getDerivative(double input) {
        return -2 * steepness * input * Math.exp(-steepness * input * input);
    }
}
