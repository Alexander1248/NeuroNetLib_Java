package ru.alexander1248.nnlib;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;

public class GeneticLearning {
    private final List<LayeredNeuralNetwork> networks;
    private double mutationCoefficient;

    public GeneticLearning(int size, LayeredNeuralNetwork example) {
        networks = new ArrayList<>(size);
        for (int i = 0; i < size; i++)
            networks.add(example.clone());
    }

    public void trainByPredicate(Predicate<LayeredNeuralNetwork> predicate) {
        int tp = -1;
        for (int i = 0; i < networks.size(); i++)
            if (predicate.test(networks.get(i))) {
                tp = i;
                break;
            }

        for (int i = 0; i < networks.size(); i++)
            if (!predicate.test(networks.get(i))) regenerateNN(i, tp);
    }
    public void trainByComparator(Comparator<LayeredNeuralNetwork> comparator) {
        networks.sort(comparator);
        for (int i = 3; i < networks.size(); i++) regenerateNN(i, (int) (Math.random() * 3));
    }

    public void regenerateNN(int i, int exampleI) {
        networks.set(i, networks.get(exampleI).clone());
        networks.get(i).mutate(mutationCoefficient);
    }

    public void setMutationCoefficient(double mutationCoefficient) {
        this.mutationCoefficient = mutationCoefficient;
    }
}
