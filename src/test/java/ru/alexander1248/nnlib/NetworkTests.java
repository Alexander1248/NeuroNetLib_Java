package ru.alexander1248.nnlib;


import junit.framework.TestCase;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class NetworkTests extends TestCase {
    private static final CalculatingType type = CalculatingType.GPU;


    public void testXOR() {
        LayeredNeuralNetwork network = new LayeredNeuralNetwork(type);
        network.initInLayer(AFunction.Sigmoid,4,2);
        network.initHiddenOrOutLayer(AFunction.Sigmoid, 2);
        network.initHiddenOrOutLayer(AFunction.Sigmoid, 1);
        network.setTrainSpeed(0.1);
        network.setMomentumCoefficient(0.9);

        double[] rr;
        double error;
        int epoch = 0;
        do {
            double x = Math.random();
            double y = Math.random();
            rr = new double[]{(int) (Math.round(x) + Math.round(y)) % 2};

            network.setInput(0, x);
            network.setInput(1, y);
            network.calculateNet();
            network.calculateError(rr);
            network.calculateNewWeights();
            error = network.getError(rr);
            epoch++;
            if (epoch % 10000 == 0) System.out.printf("Error: %1.1f\n", error);
        } while (error > 0.05);
        System.out.println();
        for (double y = 0; y < 1; y += 0.1) {
            for (double x = 0; x < 1; x += 0.1) {
                network.setInput(0, x);
                network.setInput(1, y);
                network.calculateNet();
                System.out.printf("%1.1f ",network.getOutput(0));
            }
            System.out.println();
        }
        network.calculateNet();
    }

    public void testCompressor() {
        LayeredNeuralNetwork network = new LayeredNeuralNetwork(type);
        network.initInLayer(AFunction.Sigmoid, 28 * 14, 28 * 28);
        network.initHiddenOrOutLayer(AFunction.Sigmoid, 28 * 28);
        network.setTrainSpeed(0.005);

        double error = 0;
        int epoch = 0;
        File[][] files = new File[10][];
        for (int i = 0; i < 10; i++) files[i] = new File("C:\\Projects\\JavaProjects\\NeuroNetLib\\src\\test\\resources\\dataset\\" + i).listFiles();
        do {
            int i = (int) (Math.random() * 10);
            try {
                BufferedImage img = ImageIO.read(files[i][(int) (Math.random() * files[i].length)]);
                double[] rr = new double[28 * 28];
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        network.setInput(y * 28 + x, (double) new Color(img.getRGB(x, y)).getRed() / 255);
                        rr[y * 28 + x] = (double) new Color(img.getRGB(x, y)).getRed() / 255;
                    }
                }
                network.calculateNet();
                network.calculateError(rr);
                network.calculateNewWeights();

                error = network.getError(rr);
                epoch++;
                if (epoch % 100 == 0) System.out.printf("Error: %1.1f\n", error);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } while (error > 1);

        for (int i = 0; i < 10; i++) {
            int j = (int) (Math.random() * 10);
            try {
                BufferedImage img = ImageIO.read(files[j][(int) (Math.random() * files[j].length)]);
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        network.setInput(y * 28 + x, (double) new Color(img.getRGB(x, y)).getRed() / 255);
                    }
                }
                network.calculateNet();
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        img.setRGB(x, y, new Color((float) network.getOutput(y * 28 + x), (float) network.getOutput(y * 28 + x), (float) network.getOutput(y * 28 + x)).getRGB());
                    }
                }
                ImageIO.write(img, "png", new File("C:\\Projects\\JavaProjects\\NeuroNetLib\\src\\test\\resources\\" + j + ".png"));
            } catch (IOException ignored) {
            }
        }
    }

    public void testRecognition() {
        LayeredNeuralNetwork network = new LayeredNeuralNetwork(type);
        network.initInLayer(AFunction.Sigmoid, 14 * 14, 28 * 28);
        network.initHiddenOrOutLayer(AFunction.Sigmoid, 7 * 7);
        network.initHiddenOrOutLayer(AFunction.Sigmoid, 10);
        network.setTrainSpeed(0.001);

        double error = 0;
        int epoch = 0;
        File[][] files = new File[10][];
        for (int i = 0; i < 10; i++) files[i] = new File("C:\\Projects\\JavaProjects\\NeuroNetLib\\src\\test\\resources\\dataset\\" + i).listFiles();
        do {
            int i = (int) (Math.random() * 10);
            try {
                BufferedImage img = ImageIO.read(files[i][(int) (Math.random() * files[i].length)]);
                double[] rr = new double[10];
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        network.setInput(y * 28 + x, (double) new Color(img.getRGB(x, y)).getRed() / 255);
                    }
                }
                for (int j = 0; j < 10; j++) rr[j] = j == i ? 1 : 0;

                network.calculateNet();
                network.calculateError(rr);
                network.calculateNewWeights();

                error = network.getError(rr);
                epoch++;
                if (epoch % 1000 == 0) System.out.printf("Error: %1.1f\n", error);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } while (error > 1);

        for (int i = 0; i < 10; i++) {
            int j = (int) (Math.random() * 10);
            try {
                BufferedImage img = ImageIO.read(files[j][(int) (Math.random() * files[j].length)]);
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        network.setInput(y * 28 + x, (double) new Color(img.getRGB(x, y)).getRed() / 255);
                    }
                }
                network.calculateNet();
                System.out.println(j);
                for (int k = 0; k < 10; k++) System.out.printf("%1.2f ",network.getOutput(k));
                System.out.println();
            } catch (IOException ignored) {
            }
        }
    }
}
