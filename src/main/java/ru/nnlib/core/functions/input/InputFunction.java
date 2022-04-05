package ru.nnlib.core.functions.input;

public abstract class InputFunction {
    protected double data;

    public abstract void add(double value);

    public abstract void reset();

    public double getOutput() {
        return data;
    }
}
