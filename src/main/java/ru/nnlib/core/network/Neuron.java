package ru.nnlib.core.network;

import ru.nnlib.core.functions.input.InputFunction;
import ru.nnlib.core.functions.output.ActivationFunction;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class Neuron {
    private final InputFunction inputFunction;
    private final ActivationFunction activationFunction;

    private final List<Connection> connections;

    private double output;

    public Neuron(InputFunction inputFunction, ActivationFunction activationFunction) {
        this.inputFunction = inputFunction;
        this.activationFunction = activationFunction;

        connections = new LinkedList<>();
    }

    public InputFunction getInputFunction() {
        return inputFunction;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public double getOutput() {
        return output;
    }

    public void addConnection(Connection connection) {
        connections.add(connection);
    }

    public List<Connection> getConnections() {
        return connections;
    }

    public void calculate() {
        inputFunction.reset();
        Iterator<Connection> iterator = connections.iterator();
        while (iterator.hasNext()) iterator.next().transfer();
        output = activationFunction.getOutput(inputFunction.getOutput());
    }

}
