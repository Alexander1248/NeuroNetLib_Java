package ru.nnlib.core.functions.output;

public class Step extends ActivationFunction{
    @Override
    public double getOutput(double input) {
        return input > 0 ? 1 : 0;
    }

    @Override
    public double getDerivative(double input) {
        return 0;
    }
}
