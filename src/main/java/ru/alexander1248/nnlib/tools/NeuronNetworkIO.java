package ru.alexander1248.nnlib.tools;

import ru.alexander1248.nnlib.main.AFunction;
import ru.alexander1248.nnlib.main.CalculatingType;
import ru.alexander1248.nnlib.main.LayeredNeuralNetwork;

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
            writer.write(network.getType() + " ");
            writer.write(network.getLayers().get(0).getFunction().name() + " ");
            writer.write(network.getLayers().get(0).getNeurons().length + " ");
            writer.write(network.getLayers().get(0).getInputSize() + " ");
            writer.write(network.getLayers().get(0).getRecurrent() + " ");

            for (int l = 1; l < network.getLayers().size(); l++) {
                writer.write(network.getLayers().get(l).getFunction().name() + " ");
                writer.write(network.getLayers().get(l).getNeurons().length + " ");
                writer.write(network.getLayers().get(0).getRecurrent() + " ");
            }

            for (int l = 0; l < network.getLayers().size(); l++) {
                for (int n = 0; n < network.getLayers().get(l).getNeurons().length; n++) {
                    for (int w = 0; w < network.getLayers().get(l).getNeurons()[n].getWeights().length; w++)
                        writer.write( network.getLayers().get(l).getNeurons()[n].getWeights()[w] * coef + " ");

                    writer.write(network.getLayers().get(l).getNeurons()[n].getBiasWeight() * coef + " ");
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
            network.initInLayer(AFunction.valueOf(reader.next()), reader.nextInt(), reader.nextInt(), reader.nextBoolean());
            for (int l = 1; l < network.getLayers().size(); l++)
                network.initHiddenOrOutLayer(AFunction.valueOf(reader.next()), reader.nextInt(), reader.nextBoolean());

            for (int l = 0; l < network.getLayers().size(); l++) {
                for (int n = 0; n < network.getLayers().get(l).getNeurons().length; n++) {
                    for (int w = 0; w < network.getLayers().get(l).getNeurons()[n].getWeights().length; w++)
                        network.getLayers().get(l).getNeurons()[n].getWeights()[w] = reader.nextDouble() / coef;

                    network.getLayers().get(l).getNeurons()[n].setBiasWeight(reader.nextDouble() / coef);
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
