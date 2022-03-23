package ru.alexander1248.nnlib.kernel;

import java.io.File;

public class CUDAManager {
    public static String ptxTShader = "TShader.ptx";
    public static String ptxCShader = "CShader.ptx";
   static {
//       if (!new File(ptxTShader).exists())
       ptxTShader = JCudaUtils.preparePtxFile("TShader.cu");
//       if (!new File(ptxCShader).exists())
//       ptxCShader = JCudaUtils.preparePtxFile("CShader.cu");
   }
}
