package ru.alexander1248.nnlib.kernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

public class CUDACalculating {
    private final CUfunction function;

    private final CUdeviceptr data;
    private final CUdeviceptr links;
    private final CUdeviceptr weights;
    private final CUdeviceptr weightedSum;

    int len;

    public CUDACalculating(int prevLength) {
        len = prevLength;
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, CUDAManager.ptxCShader);

        function = new CUfunction();
        cuModuleGetFunction(function, module, "train");

        data = new CUdeviceptr();
        cuMemAlloc(data, (long) prevLength * Sizeof.DOUBLE);

        weights = new CUdeviceptr();
        cuMemAlloc(weights, (long) prevLength * Sizeof.DOUBLE);

        weightedSum = new CUdeviceptr();
        cuMemAlloc(weightedSum, (long) prevLength * Sizeof.DOUBLE);

        links = new CUdeviceptr();
        cuMemAlloc(links, (long) prevLength * Sizeof.INT);
    }

    public void run(double[] data, double[] weights, int[] links, double[] weightedSum) {
        cuMemcpyHtoD(this.data, Pointer.to(data), (long) data.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.weights, Pointer.to(weights), (long) weights.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(this.links, Pointer.to(links), (long) links.length * Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new double[]{len}),
                Pointer.to(this.weightedSum),
                Pointer.to(this.data),
                Pointer.to(this.weights),
                Pointer.to(this.links)
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

        cuMemcpyDtoH(Pointer.to(weightedSum), this.weightedSum, (long) data.length * Sizeof.DOUBLE);
        cuMemcpyDtoH(Pointer.to(weights), this.weights, (long) data.length * Sizeof.DOUBLE);

    }

    public void destroyStream() {

    }

    @Override
    protected void finalize() throws Throwable {
        destroyStream();
        super.finalize();
    }
}
