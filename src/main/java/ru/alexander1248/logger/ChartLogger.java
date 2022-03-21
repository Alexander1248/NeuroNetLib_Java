package ru.alexander1248.logger;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

public class ChartLogger {
    private FileWriter writer;

    public ChartLogger(String filename, String xName, String yName) {
        try {
            writer = new FileWriter(filename + ".csv");
            writer.write(xName + "," + yName + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeData(double i, double value) {
        try {
            writer.write(i + "," + value + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void writeData(int i, int value) {
        try {
            writer.write(i + "," + value + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void close() {
        try {
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
