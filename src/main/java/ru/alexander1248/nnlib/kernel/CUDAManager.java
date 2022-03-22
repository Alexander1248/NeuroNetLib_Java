package ru.alexander1248.nnlib.kernel;

import jcuda.driver.JCudaDriver;

public class CUDAManager {
    public static String ptxTShader;
    public static String ptxCShader;
    static {
        JCudaDriver.setExceptionsEnabled(true);
        ptxTShader = JCudaUtils.preparePtxFile("TShader.cu");
        ptxCShader = JCudaUtils.preparePtxFile("CShader.cu");
    }
}
