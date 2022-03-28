package ru.nnlib.core.functions;

public class Sigmoid extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return 1 / (1 + Math.exp(steepness - shift * input));
    }

    @Override
    public double getDerivative(double input) {
        double a = Math.exp(shift - steepness * input);
        return steepness * a / (a * a + 2 * a + 1);
    }
}
