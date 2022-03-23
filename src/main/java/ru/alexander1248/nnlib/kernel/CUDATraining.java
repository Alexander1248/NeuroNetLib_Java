package ru.alexander1248.nnlib.kernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;


import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

public class CUDATraining {
    private final CUfunction function;

    private final CUdeviceptr data;
    private final CUdeviceptr acceleration;
    private final CUdeviceptr weights;
    private final CUdeviceptr links;
    private final CUdeviceptr error;

    public CUDATraining(int length, int prevLength) {
        JCudaDriver.setExceptionsEnabled(true);
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, CUDAManager.ptxTShader);

        function = new CUfunction();
        cuModuleGetFunction(function, module, "train");


        data = new CUdeviceptr();
        cuMemAlloc(data, (long) prevLength * Sizeof.DOUBLE);

        acceleration = new CUdeviceptr();
        cuMemAlloc(acceleration, (long) length * prevLength * Sizeof.DOUBLE);

        weights = new CUdeviceptr();
        cuMemAlloc(weights, (long) length * prevLength * Sizeof.DOUBLE);

        links = new CUdeviceptr();
        cuMemAlloc(links, (long) length * prevLength * Sizeof.INT);

        error = new CUdeviceptr();
        cuMemAlloc(error, (long) length * Sizeof.DOUBLE);
    }

    public void run(double[] data, double[] acceleration, double[] weights, int[] links, double[] error, double trainSpeed, double momentumCoefficient) {
        cuMemcpyHtoD(this.data, Pointer.to(data), (long) data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.acceleration, Pointer.to(acceleration), (long) acceleration.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.weights, Pointer.to(weights), (long) weights.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.links, Pointer.to(links), (long) links.length * Sizeof.INT);
        cuMemcpyHtoD(this.error, Pointer.to(error), (long) error.length * Sizeof.DOUBLE);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{error.length}),
                Pointer.to(new int[]{data.length}),
                Pointer.to(this.data),
                Pointer.to(this.acceleration),
                Pointer.to(this.weights),
                Pointer.to(this.links),
                Pointer.to(this.error),
                Pointer.to(new double[]{trainSpeed}),
                Pointer.to(new double[]{momentumCoefficient})
        );

        int blockSizeX = Math.min(256, data.length);
        int gridSizeX = (int)Math.ceil((double)data.length / blockSizeX);
        int blockSizeY = Math.min(256, error.length);
        int gridSizeY = (int)Math.ceil((double)error.length / blockSizeX);
         cuLaunchKernel(function,
                gridSizeX,  gridSizeY, 1,      // Grid dimension
                blockSizeX, blockSizeY, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
         cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(acceleration), this.acceleration, (long) acceleration.length * Sizeof.DOUBLE);
        cuMemcpyDtoH(Pointer.to(weights), this.weights, (long) weights.length * Sizeof.DOUBLE);
    }

    public void destroyStream() {
        cuMemFree(data);
        cuMemFree(acceleration);
        cuMemFree(weights);
        cuMemFree(links);
    }

    @Override
    protected void finalize() throws Throwable {
        destroyStream();
        super.finalize();
    }
}
