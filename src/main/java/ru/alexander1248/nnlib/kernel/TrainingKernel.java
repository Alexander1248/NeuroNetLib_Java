package ru.alexander1248.nnlib.kernel;


import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class TrainingKernel {

    public TrainingKernel() {

        
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "NNTrain.ptx");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");
    }

    public void run() {

    }
}
