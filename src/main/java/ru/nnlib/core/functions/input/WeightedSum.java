package ru.nnlib.core.functions.input;

public class WeightedSum extends InputFunction{
    @Override
    public void add(double value) {
        data += value;
    }

    @Override
    public void reset() {
        data = 0;
    }
}
