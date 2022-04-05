package ru.nnlib.core.functions.output;

public class ReLU extends ActivationFunction{
    @Override
    public double getOutput(double input) {
        return Math.max(0, steepness * input);
    }

    @Override
    public double getDerivative(double input) {
        return steepness * input > 0 ? steepness : 0;
    }
}
