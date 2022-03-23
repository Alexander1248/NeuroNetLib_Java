package ru.nnlib.tools;

public class NNHelpingTools {
    private NNHelpingTools() {}

    public static double trainingSpeedCalculator(double maxTSLimit, double forceCoefficient, double errorDelta) {
        return maxTSLimit / (1 + Math.exp(1 - forceCoefficient * errorDelta));
    }
}
