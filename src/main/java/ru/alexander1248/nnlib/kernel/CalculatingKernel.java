package ru.alexander1248.nnlib.kernel;


import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.*;

public class CalculatingKernel {

    public CalculatingKernel() {

        
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "CudaNNCalculating.ptx");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");
    }

    public void run() {

    }
}
