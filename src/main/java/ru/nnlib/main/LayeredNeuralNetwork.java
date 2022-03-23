package ru.nnlib.main;

import java.util.ArrayList;
import java.util.List;

public class LayeredNeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    double trainSpeed = 0.1, momentumCoefficient = 0;

    private final CalculatingType type;

    public LayeredNeuralNetwork(CalculatingType type) {
        this.type = type;
    }
    public LayeredNeuralNetwork(CalculatingType type, AFunction[] layersFunctions, int[] layersSizes, boolean[] reccurent) {
        this(type);
        initInLayer(layersFunctions[0], layersSizes[1], layersSizes[0], reccurent[0]);
        for (int i = 1; i < layers.size(); i++) initHiddenOrOutLayer(layersFunctions[i], layersSizes[i + 1], reccurent[i]);
    }
    public LayeredNeuralNetwork(CalculatingType type, AFunction[] layersFunctions, int[] layersSizes) {
        this(type);
        initInLayer(layersFunctions[0], layersSizes[1], layersSizes[0], false);
        for (int i = 1; i < layers.size(); i++) initHiddenOrOutLayer(layersFunctions[i], layersSizes[i + 1], false);
    }

    public void initHiddenOrOutLayer(AFunction function, int size) {
        layers.add(new Layer(function, layers.get(layers.size() - 1), size, false, type));
    }
    public void initHiddenOrOutLayer(AFunction function, int size, boolean reccurent) {
        layers.add(new Layer(function, layers.get(layers.size() - 1), size, reccurent, type));
    }
    public void initHiddenOrOutLayer(AFunction function, boolean[][] links) {
        layers.add(new Layer(function, layers.get(layers.size() - 1), links, false, type));
    }
    public void initHiddenOrOutLayer(AFunction function, boolean[][] links, boolean reccurent) {
        layers.add(new Layer(function, layers.get(layers.size() - 1), links, reccurent, type));
    }

    public void initInLayer(AFunction function, boolean[][] links) {
        layers.add(new Layer(function, links, false, type));
    }
    public void initInLayer(AFunction function, boolean[][] links, boolean reccurent) {
        layers.add(new Layer(function, links, reccurent, type));
    }
    public void initInLayer(AFunction function, int size, int inputSize) {
        layers.add(new Layer(function, inputSize, size, false, type));
    }
    public void initInLayer(AFunction function, int size, int inputSize, boolean reccurent) {
        layers.add(new Layer(function, inputSize, size, reccurent, type));
    }

    public void setInput(int i,double value) {
        layers.get(0).setInput(i,value);
    }

    public double getOutput(int i) {
        return layers.get(layers.size() - 1).getOutput(i);
    }
    public double getOutput(int layer,int i) {
        return layers.get(layer).getOutput(i);
    }

    public void calculateNet() {
        for (Layer layer : layers) layer.calculateLayer();
    }

    public void calculateError(double[] rightResults) {
        layers.get(layers.size() - 1).calculateOutLayerError(rightResults);
        for (int i = layers.size() - 2; i >= 0; i--) layers.get(i).calculateInOrHiddenLayerError(layers.get(i + 1));
    }

    public void calculateNewWeights() {
        for (int i = 1; i < layers.size(); i++) layers.get(i).calculateNewWeights(trainSpeed, momentumCoefficient);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double getError(double[] rightResults) {
        double error = 0;
        for (int i = 0; i < rightResults.length; i++) error += Math.abs(rightResults[i] - getOutput(i));
        return error / rightResults.length * 100;
    }


    public void setTrainSpeed(double trainSpeed) {
        this.trainSpeed = trainSpeed;
    }

    public void setMomentumCoefficient(double momentumCoefficient) {
        this.momentumCoefficient = momentumCoefficient;
    }


    public void mutate(double coefficient) {
        for (int i = 0; i < 10; i++)
            layers.get((int) (Math.random() * layers.size())).mutate(coefficient / 10);
    }

    public LayeredNeuralNetwork clone() {
        LayeredNeuralNetwork network = new LayeredNeuralNetwork(type);
        network.initInLayer(layers.get(0).getFunction(), layers.get(0).l, layers.get(0).getRecurrent());
        for (int l = 1; l < layers.size(); l++) {
            network.initHiddenOrOutLayer(layers.get(l).getFunction(), layers.get(l).l, layers.get(0).getRecurrent());
            for (int n = 0; n < layers.get(l).getLength(); n++) {
                System.arraycopy(layers.get(l).getWeights()[n], 0, network.getLayers().get(l).getWeights()[n], 0, layers.get(l).getWeights().length);
                network.getLayers().get(l).getBiasWeight()[n] = layers.get(l).getBiasWeight()[n];
            }
        }
        network.setTrainSpeed(trainSpeed);
        network.setMomentumCoefficient(momentumCoefficient);
        return network;
    }

    public CalculatingType getType() {
        return type;
    }
}
