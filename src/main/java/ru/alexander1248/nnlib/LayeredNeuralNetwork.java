package ru.alexander1248.nnlib;

import java.util.ArrayList;
import java.util.List;

public class LayeredNeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    private final CalculatingType type;
    double trainSpeed = 0.1, momentumCoefficient = 0;


    public LayeredNeuralNetwork(CalculatingType type) {
        this.type = type;
    }
    public LayeredNeuralNetwork(CalculatingType type, AFunction[] layersFunctions, int[] layersSizes) {
        this(type);
        initInLayer(layersFunctions[0], layersSizes[1], layersSizes[0]);
        for (int i = 1; i < layers.size(); i++) initHiddenOrOutLayer(layersFunctions[i], layersSizes[i + 1]);
    }

    public void initHiddenOrOutLayer(AFunction function, int size) {
        Layer layer = new Layer(layers.get(layers.size() - 1),function,size);
        layer.setCalculatingType(type);
        layers.add(layer);
    }
    public void initInLayer(AFunction function, int size, int inputSize){
        Layer layer = new Layer(inputSize,function,size);
        layer.setCalculatingType(type);
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
        for (int i = 1; i < layers.size(); i++) layers.get(i).calculateNewWeights(trainSpeed,momentumCoefficient);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double getError(double[] rightResults) {
        double error = 0;
        for (int i = 0; i < rightResults.length; i++) error += Math.abs(rightResults[i] - getOutput(i));
        return error / rightResults.length * 100;
    }

    public CalculatingType getType() {
        return type;
    }

    public void setTrainSpeed(double trainSpeed) {
        this.trainSpeed = trainSpeed;
    }

    public void setMomentumCoefficient(double momentumCoefficient) {
        this.momentumCoefficient = momentumCoefficient;
    }
}
