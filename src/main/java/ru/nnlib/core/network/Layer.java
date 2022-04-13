package ru.nnlib.core.network;

import org.jocl.*;

public class Layer {
    private float[] input;
    private float[] weights;
    private float[] output;

    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_kernel kernel;
    private cl_program program;
    private cl_mem[] memObjects;

    private static final String programSource = "__kernel void "+
            "sampleKernel(__global const float *input,"+
            "             __global const float *weights,"+
            "             __global float *output," +
            "             __global float size)"+
            "{"+
            "    int x = get_global_id(0);" +
            "    int y = get_global_id(1);" +
            "    int i = y * size + x;"+
            "    output[y] += input[x] + weights[i];"+
            "}";

    public Layer(int inputSize, int outputSize) {
        input = new float[inputSize];
        weights = new float[inputSize * outputSize];
        output = new float[outputSize];
    }

    public void setInputs(float[] input) {
        this.input = input;
    }

    public float[] getOutputs() {
        return output;
    }

    public void buildThread() {
        final int platformIndex = 0;
        final long deviceType = CL.CL_DEVICE_TYPE_GPU;
        final int deviceIndex = 0;

        Pointer srcIn = Pointer.to(input);
        Pointer srcOut = Pointer.to(output);

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int[] numPlatformsArray = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = CL.clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        commandQueue =
                CL.clCreateCommandQueue(context, device, 0, null);

        // Allocate the memory objects for the input and output data
        cl_mem memObjects[] = new cl_mem[3];
        memObjects[0] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * input.length, srcIn, null);
        memObjects[1] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * output.length, srcOut, null);
        memObjects[2] = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_WRITE,
                (long) Sizeof.cl_float * input.length * output.length, null, null);

        // Create the program from the source code
        program = CL.clCreateProgramWithSource(context,
                1, new String[]{ programSource }, null, null);
        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        kernel = CL.clCreateKernel(program, "", null);
    }

    public void calculate() {
        // Set the arguments for the kernel
        CL.clSetKernelArg(kernel, 0,
                Sizeof.cl_mem, Pointer.to(memObjects[0]));
        CL.clSetKernelArg(kernel, 1,
                Sizeof.cl_mem, Pointer.to(memObjects[1]));
        CL.clSetKernelArg(kernel, 2,
                Sizeof.cl_mem, Pointer.to(memObjects[2]));
        CL.clSetKernelArg(kernel, 2,
                Sizeof.cl_mem, Pointer.to(new int[] {input.length}));

        // Set the work-item dimensions
        long[] global_work_size = new long[]{input.length, output.length};
        long[] local_work_size = new long[]{1};

        // Execute the kernel
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        CL.clEnqueueReadBuffer(commandQueue, memObjects[2], CL.CL_TRUE, 0,
                (long) output.length * Sizeof.cl_float, Pointer.to(weights), 0, null, null);

    }

    public void releaseThread() {
        CL.clReleaseMemObject(memObjects[0]);
        CL.clReleaseMemObject(memObjects[1]);
        CL.clReleaseMemObject(memObjects[2]);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);
    }
}
