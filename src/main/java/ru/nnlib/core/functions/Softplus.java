package ru.nnlib.core.functions;

public class Softplus extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return Math.log(1 + Math.exp(steepness * input));
    }

    @Override
    public double getDerivative(double input) {
        double e = Math.exp(steepness * input);
        return steepness * e / (1 + e);
    }
}
