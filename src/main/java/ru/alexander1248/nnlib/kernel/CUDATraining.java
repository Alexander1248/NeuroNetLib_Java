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

    public CUDATraining(int length) {
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


        data = new CUdeviceptr();
        cuMemAlloc(data, (long) length * Sizeof.DOUBLE);

        acceleration = new CUdeviceptr();
        cuMemAlloc(acceleration, (long) length * Sizeof.DOUBLE);

        weights = new CUdeviceptr();
        cuMemAlloc(weights, (long) length * Sizeof.DOUBLE);

        links = new CUdeviceptr();
        cuMemAlloc(data, (long) length * Sizeof.INT);
    }

    public void run(double[] data, double[] acceleration, double[] weights, int[] links, double error, double trainSpeed, double momentumCoefficient) {
        cuMemcpyHtoD(this.data, Pointer.to(data), (long) data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.acceleration, Pointer.to(acceleration), (long) data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.weights, Pointer.to(weights), (long) data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.links, Pointer.to(links), (long) data.length * Sizeof.DOUBLE);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new double[]{data.length}),
                Pointer.to(this.data),
                Pointer.to(this.acceleration),
                Pointer.to(this.weights),
                Pointer.to(this.links),
                Pointer.to(new double[]{error}),
                Pointer.to(new double[]{trainSpeed}),
                Pointer.to(new double[]{momentumCoefficient})
        );
        int blockSizeX = Math.min(256, data.length);
        int gridSizeX = (int)Math.ceil((double)data.length / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(acceleration), this.acceleration, (long) data.length * Sizeof.DOUBLE);
        cuMemcpyDtoH(Pointer.to(weights), this.weights, (long) data.length * Sizeof.DOUBLE);
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
