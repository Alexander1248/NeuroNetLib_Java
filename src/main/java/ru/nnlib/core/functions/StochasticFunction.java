package ru.nnlib.core.functions;

public record StochasticFunction(ActivationFunction function) {

    public double getOutput(double input) {
        return function.getOutput(input) > Math.random() ? 1 : 0;
    }

    public double getDerivative(double input) {
        return function.getDerivative(input);
    }
}
