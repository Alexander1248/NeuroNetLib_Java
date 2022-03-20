package ru.alexander1248.nnlib;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Scanner;

public class NeuronNetworkIO {
    private static final double coef = 1000000;

    public static void write(String filename, LayeredNeuralNetwork network) {
        try {
            FileWriter writer = new FileWriter(filename + ".nwd");
            writer.write(network.getType().name() + " ");
            writer.write(network.getLayers().get(0).getFunction().name() + " ");
            writer.write(network.getLayers().get(0).getSize() + " ");
            writer.write(network.getLayers().get(0).getInputSize() + " ");

            for (int l = 1; l < network.getLayers().size(); l++) {
                writer.write(network.getLayers().get(l).getFunction().name() + " ");
                writer.write(network.getLayers().get(l).getSize() + " ");
            }

            for (int l = 0; l < network.getLayers().size(); l++) {
                for (int n = 0; n < network.getLayers().get(l).getSize(); n++) {
                    for (int w = 0; w < network.getLayers().get(l).weights[n].length; w++)
                        writer.write( network.getLayers().get(l).weights[n][w] * coef + " ");

                    writer.write(network.getLayers().get(l).biasWeight[n] * coef + " ");
                }
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static LayeredNeuralNetwork read(String filename) {
        try {
            Scanner reader = new Scanner(new File(filename + ".nwd"));
            reader.useLocale(Locale.UK);
            LayeredNeuralNetwork network = new LayeredNeuralNetwork(CalculatingType.valueOf(reader.next()));
            network.initInLayer(AFunction.valueOf(reader.next()), reader.nextInt(), reader.nextInt());
            for (int l = 1; l < network.getLayers().size(); l++)
                network.initHiddenOrOutLayer(AFunction.valueOf(reader.next()), reader.nextInt());

            for (int l = 0; l < network.getLayers().size(); l++) {
                for (int n = 0; n < network.getLayers().get(l).weights.length; n++) {
                    for (int w = 0; w < network.getLayers().get(l).weights[n].length; w++)
                        network.getLayers().get(l).weights[n][w] = reader.nextDouble() / coef;

                    network.getLayers().get(l).biasWeight[n] = reader.nextDouble() / coef;
                }
            }
            reader.close();
            return network;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

}
