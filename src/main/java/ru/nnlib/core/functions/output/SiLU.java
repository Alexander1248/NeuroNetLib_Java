package ru.nnlib.core.functions.output;

public class SiLU extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return steepness * input / (1 + Math.exp(-steepness * input));
    }

    @Override
    public double getDerivative(double input) {
        return getOutput(input) + (1 - getOutput(input)) / (1 + Math.exp(-steepness * input));
    }
}
