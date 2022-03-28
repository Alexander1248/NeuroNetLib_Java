package ru.nnlib.core.functions;

public abstract class ActivationFunction {
    public double steepness = 1;
    public double shift = 0;

    public abstract double getOutput(double input);
    public abstract double getDerivative(double input);
}
