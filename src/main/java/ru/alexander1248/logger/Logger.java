package ru.alexander1248.logger;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger {
    static private final SimpleDateFormat formatDate = new SimpleDateFormat("dd.MM.yyyy HH:mm:ss:S");
    private static boolean append = false;

    private Logger() {}

    static public void append(boolean appending) {
        append = appending;
    }
    static public void i(String msg) {
        try (FileWriter log = new FileWriter("log.txt",append)) {
            Date date = new Date();

            log.write(  formatDate.format(date) + " INFO - " + msg + "\r\n");
        } catch (IOException ignored) {}
    }
    static public void w(String msg) {
        try (FileWriter log = new FileWriter("log.txt",append)) {
            Date date = new Date();

            log.write(formatDate.format(date) + " WARN - " + msg + "\r\n");
        } catch (IOException ignored) {}
    }
    static public void e(String msg) {
        try (FileWriter log = new FileWriter("log.txt",append)) {
            Date date = new Date();

            log.write(formatDate.format(date) + " ERROR - " + msg + "\r\n");
        } catch (IOException ignored) {}
    }
}
