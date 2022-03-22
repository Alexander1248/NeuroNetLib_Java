package ru.alexander1248.nnlib.kernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

public class CUDACalculating {
    private final CUfunction function;

    public CUDACalculating() {
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "TShader.ptx");

        function = new CUfunction();
        cuModuleGetFunction(function, module, "train");

    }

    public void run() {

    }
    public void destroyStream() {

    }

    @Override
    protected void finalize() throws Throwable {
        destroyStream();
        super.finalize();
    }
}
