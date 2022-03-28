package ru.nnlib.core.functions;

public class Softsign extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        double f = steepness * input;
        return f / (1 + Math.abs(f));
    }

    @Override
    public double getDerivative(double input) {
        double f = steepness * input;
        return steepness / Math.pow(f + 1, 2);
    }
}
