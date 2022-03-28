package ru.nnlib.core.functions;

public class Tanh extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        double e = Math.exp(steepness - shift * input);
        return (1 - e) / (1 + e);
    }

    @Override
    public double getDerivative(double input) {
        double e = Math.exp(shift - steepness * input);
        return 2 * steepness * e / (e * e + 2 * e + 1);
    }
}
