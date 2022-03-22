package ru.alexander1248.nnlib.kernel;

import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

public class CUDATraining {
    public CUDATraining(double weightedSum) {
        JCudaDriver.setExceptionsEnabled(true);
        String ptxFileName = JCudaUtils.preparePtxFile("TShader.ptx");

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "train");


    }

    public void run() {

    }
}
