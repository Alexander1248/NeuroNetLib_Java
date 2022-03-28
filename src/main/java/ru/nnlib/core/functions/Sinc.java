package ru.nnlib.core.functions;

public class Sinc extends ActivationFunction {
    @Override
    public double getOutput(double input) {
        return Math.sin(steepness * input) / steepness / input;
    }

    @Override
    public double getDerivative(double input) {
        return Math.cos(steepness * input) / input - Math.sin(steepness * input) / steepness / input / input;
    }
}
