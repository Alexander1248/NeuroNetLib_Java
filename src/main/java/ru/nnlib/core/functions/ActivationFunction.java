package ru.nnlib.core.functions;

public abstract class ActivationFunction {
    public abstract double getOutput(double input);
    public abstract double getDerivative(double input);
}
