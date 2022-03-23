package ru.alexander1248.nnlib.tools;

import java.io.FileWriter;
import java.io.IOException;

public class TableLogger {
    private FileWriter writer;

    public TableLogger(String filename, String xName, String yName) {
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
