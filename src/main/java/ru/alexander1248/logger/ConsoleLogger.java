package ru.alexander1248.logger;

import java.text.SimpleDateFormat;
import java.util.Date;

public class ConsoleLogger {
    static private final SimpleDateFormat formatDate = new SimpleDateFormat("dd.MM.yyyy HH:mm:ss:S");

    private ConsoleLogger() {}
    static public void i(String msg) {
        Date date = new Date();
        System.out.println(formatDate.format(date) + " INFO - " + msg);
    }
    static public void w(String msg) {
        Date date = new Date();
        System.out.println(formatDate.format(date) + " WARN - " + msg);
    }
    static public void e(String msg) {
        Date date = new Date();
        System.out.println(formatDate.format(date) + " ERROR - "  + msg);
    }
    static public void ui(String msg) {
        Date date = new Date();
        System.out.print("\r" + formatDate.format(date) + " INFO - " + msg);
    }
    static public void uw(String msg) {
        Date date = new Date();
        System.out.print("\r" + formatDate.format(date) + " WARN - " + msg);
    }
    static public void ue(String msg) {
        Date date = new Date();
        System.out.print("\r" + formatDate.format(date) + " ERROR - " + msg);
    }
    static public void skip() {
        System.out.println();
    }
}
