package ru.alexander1248.nnlib.main;

import java.util.ArrayList;
import java.util.List;

public class LayeredNeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    double trainSpeed = 0.1, momentumCoefficient = 0;


    public LayeredNeuralNetwork() {}
    public LayeredNeuralNetwork(AFunction[] layersFunctions, int[] layersSizes, boolean[] reccurent) {
        initInLayer(layersFunctions[0], layersSizes[1], layersSizes[0], reccurent[0]);
        for (int i = 1; i < layers.size(); i++) initHiddenOrOutLayer(layersFunctions[i], layersSizes[i + 1], reccurent[i]);
    }
    public LayeredNeuralNetwork(AFunction[] layersFunctions, int[] layersSizes) {
        initInLayer(layersFunctions[0], layersSizes[1], layersSizes[0], false);
        for (int i = 1; i < layers.size(); i++) initHiddenOrOutLayer(layersFunctions[i], layersSizes[i + 1], false);
    }

    public void initHiddenOrOutLayer(AFunction function, int size) {
        Layer layer = new Layer(function, layers.get(layers.size() - 1), size, false);
        layers.add(layer);
    }
    public void initHiddenOrOutLayer(AFunction function, int size, boolean reccurent) {
        Layer layer = new Layer(function, layers.get(layers.size() - 1), size, reccurent);
        layers.add(layer);
    }
    public void initHiddenOrOutLayer(AFunction function, boolean[][] links) {
        Layer layer = new Layer(function, layers.get(layers.size() - 1), links, false);
        layers.add(layer);
    }
    public void initHiddenOrOutLayer(AFunction function, boolean[][] links, boolean reccurent) {
        Layer layer = new Layer(function, layers.get(layers.size() - 1), links, reccurent);
        layers.add(layer);
    }

    public void initInLayer(AFunction function, boolean[][] links){
        Layer layer = new Layer(function, links, false);
        layers.add(layer);
    }
    public void initInLayer(AFunction function, boolean[][] links, boolean reccurent){
        Layer layer = new Layer(function, links, reccurent);
        layers.add(layer);
    }
    public void initInLayer(AFunction function, int size, int inputSize){
        Layer layer = new Layer(function, inputSize, size, false);
        layers.add(layer);
    }
    public void initInLayer(AFunction function, int size, int inputSize, boolean reccurent){
        Layer layer = new Layer(function, inputSize, size, reccurent);
        layers.add(layer);
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
        LayeredNeuralNetwork network = new LayeredNeuralNetwork();
        network.initInLayer(layers.get(0).getFunction(), layers.get(0).getNeurons().length, layers.get(0).getInputSize(), layers.get(0).getRecurrency());
        for (int l = 1; l < layers.size(); l++) {
            network.initInLayer(layers.get(l).getFunction(), layers.get(l).getNeurons().length, layers.get(l).getInputSize(), layers.get(0).getRecurrency());
            for (int n = 0; n < layers.get(l).getNeurons().length; n++) {
                for (int w = 0; w < layers.get(l).getNeurons()[n].weights.length; w++)
                    network.getLayers().get(l).getNeurons()[n].weights[w] = layers.get(l).getNeurons()[n].weights[w];
                network.getLayers().get(l).getNeurons()[n].biasWeight = layers.get(l).getNeurons()[n].biasWeight;
            }
        }
        network.setTrainSpeed(trainSpeed);
        network.setMomentumCoefficient(momentumCoefficient);
        return network;
    }
}
