package ru.nnlib.core.network;


public class Connection {
    private final Neuron from;
    private final Neuron to;
    private double weight;

    public Connection(Neuron from, Neuron to) {
        this.from = from;
        this.to = to;
        weight = Math.random() - 0.5;
    }

    public void transfer() {
        to.getInputFunction().add(from.getOutput() * weight);
    }

    public void editWeight(double delta) {
        weight += delta;
    }

    public double getWeight() {
        return weight;
    }
}
