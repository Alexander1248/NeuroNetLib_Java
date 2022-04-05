package ru.nnlib.core.functions.output;

public class Identity extends ActivationFunction{
    @Override
    public double getOutput(double input) {
        return input * steepness;
    }

    @Override
    public double getDerivative(double input) {
        return steepness;
    }
}
