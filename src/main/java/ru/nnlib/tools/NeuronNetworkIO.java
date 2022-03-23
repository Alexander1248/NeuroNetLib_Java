package ru.nnlib.tools;

import ru.nnlib.core.AFunction;
import ru.nnlib.kernel.CalculatingType;
import ru.nnlib.core.LayeredNeuralNetwork;

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
            writer.write(network.getLayers().get(0).getLength() + " ");
            writer.write(network.getLayers().get(0).getInputSize() + " ");
            writer.write(network.getLayers().get(0).getRecurrent() + " ");

            for (int l = 1; l < network.getLayers().size(); l++) {
                writer.write(network.getLayers().get(l).getFunction().name() + " ");
                writer.write(network.getLayers().get(l).getLength() + " ");
                writer.write(network.getLayers().get(0).getRecurrent() + " ");
            }

            for (int l = 0; l < network.getLayers().size(); l++) {
                for (int w = 0; w < network.getLayers().get(l).getWeights().length; w++)
                    writer.write( network.getLayers().get(l).getWeights()[w] * coef + " ");
                for (int n = 0; n < network.getLayers().get(l).getLength(); n++)
                    writer.write(network.getLayers().get(l).getBiasWeight()[n] * coef + " ");

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
                for (int w = 0; w < network.getLayers().get(l).getWeights().length; w++)
                    network.getLayers().get(l).getWeights()[w] = reader.nextDouble() / coef;
                for (int n = 0; n < network.getLayers().get(l).getLength(); n++)
                    network.getLayers().get(l).getBiasWeight()[n] = reader.nextDouble() / coef;

            }
            reader.close();
            return network;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

}
