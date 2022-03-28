package ru.nnlib.core.functions;

public class LeakyReLU extends ActivationFunction{
    @Override
    public double getOutput(double input) {
        return steepness * input > 0 ? steepness * input : (steepness * input) / 100;
    }

    @Override
    public double getDerivative(double input) {
        return steepness * input > 0 ? steepness : steepness / 100;
    }
}
