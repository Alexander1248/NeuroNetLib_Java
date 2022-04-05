package ru.nnlib.core.network;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class Layer {
    private final List<Neuron> neurons = new LinkedList<>();

    public void addNeuron(Neuron neuron) {
        neurons.add(neuron);
    }

    public void calculate() {
        Iterator<Neuron> iterator = neurons.iterator();
        while (iterator.hasNext()) iterator.next().calculate();
    }
}
