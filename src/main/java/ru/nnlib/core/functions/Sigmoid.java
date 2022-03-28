package ru.nnlib.core.functions;

public class Sigmoid extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return 1 / (1 + Math.exp(-steepness * input));
    }

    @Override
    public double getDerivative(double input) {
        double a = Math.exp(-steepness * input);
        return steepness * a / Math.pow(a + 1, 2);
    }
}
