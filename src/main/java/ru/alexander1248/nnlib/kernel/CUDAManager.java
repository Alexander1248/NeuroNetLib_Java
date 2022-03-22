package ru.alexander1248.nnlib.kernel;

import jcuda.driver.JCudaDriver;

public class CUDAManager {
    public static String ptxTShader;
    public static String ptxCShader;
    public static void run() {
        ptxTShader = JCudaUtils.preparePtxFile("TShader.cu");
        ptxCShader = JCudaUtils.preparePtxFile("CShader.cu");
    }
}
