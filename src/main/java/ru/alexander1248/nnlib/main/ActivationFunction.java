package ru.alexander1248.nnlib.main;

public class ActivationFunction {
    private ActivationFunction() {}


    static public double GetFunction(AFunction type, double weightedSum) {
        switch (type)
        {
            case Identity: return weightedSum;

            case Sigmoid: return 1.0 / (1.0 + Math.exp(-weightedSum));

            case Tangent: return (1 - Math.exp(-2.0 * weightedSum)) / (1 + Math.exp(-2.0 * weightedSum));

            case Softsign: return weightedSum / (1 + Math.abs(weightedSum));

            case ReLU: return Math.max(0, weightedSum);

            case LeakyReLU: return Math.max(0.01 * weightedSum, weightedSum);

            case SiLU: return weightedSum / (1 + Math.exp(-weightedSum));

            case ELU:
                if (weightedSum <= 0) return Math.exp(weightedSum) - 1;
                else return weightedSum;

            case SoftPlus: return Math.log(1 + Math.exp(weightedSum));

            case Sinc: return Math.sin(weightedSum) / weightedSum;

            case Gaussian: return Math.exp(-weightedSum * weightedSum);

            case NCU: return weightedSum - Math.pow(weightedSum, 3);

            case SQU: return weightedSum * weightedSum + weightedSum;

            case GCU: return weightedSum * Math.cos(weightedSum);

            default: return -1;
        }
    }
    static public double GetDerivative(AFunction type, double weightedSum) {

        switch (type) {
            case Identity: return 1;

            case Sigmoid: return GetFunction (type, weightedSum) *(1 - GetFunction (type, weightedSum));

            case Tangent:
                double buff = GetFunction (type, weightedSum);
                return 1 - buff * buff;

            case Softsign: return weightedSum / Math.pow(1 +  Math.abs(weightedSum), 2);

            case ReLU:
                if (weightedSum < 0) return 0;
                else return 1;

            case LeakyReLU:
                if (weightedSum < 0) return 0.01;
                else return 1;

            case SiLU: return GetFunction (type, weightedSum)+ GetFunction(AFunction.Sigmoid, weightedSum) * (1 - GetFunction (type, weightedSum));

            case ELU:
                if (weightedSum < 0) return  Math.exp(weightedSum);
                else return 1;

            case SoftPlus: return GetFunction (AFunction.Sigmoid, weightedSum);

            case Sinc:
                if (weightedSum == 0) return 0;
                else return  Math.cos(weightedSum) / weightedSum -  Math.sin(weightedSum) /  Math.pow(weightedSum, 2);

            case Gaussian: return -2 * weightedSum *  Math.exp(-weightedSum * weightedSum);

            case NCU: return 1 - 3 *  Math.pow(weightedSum, 2);

            case SQU: return 2 * weightedSum + 1;

            case GCU: return  Math.cos(weightedSum) - weightedSum *  Math.sin(weightedSum);

            default: return -1;
        }
    }

}
