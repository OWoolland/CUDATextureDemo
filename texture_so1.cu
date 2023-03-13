#include <stdio.h>
#include <stdint.h>

typedef float4 mt;  // use an integer type

__global__ void kernel(cudaTextureObject_t tex)
{
    float x = 0.5;
    float y = 0.5;
    mt val = tex2D<mt>(tex, x, y);
    printf("%f, ", val.x);
}

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("texturePitchAlignment: %lu\n", prop.texturePitchAlignment);
    cudaTextureObject_t tex;
    const int num_rows = 4;
    const int num_cols = prop.texturePitchAlignment*1; // should be able to use a different multiplier here
    const int ts = num_cols*num_rows;
    const int ds = ts*sizeof(mt);
    mt dataIn[ds];
    for (int i = 0; i < ts; i++) dataIn[i] = float4(0,1,2,3);
    mt* dataDev = 0;
    cudaMalloc((void**)&dataDev, ds);
    cudaMemcpy(dataDev, dataIn, ds, cudaMemcpyHostToDevice);
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width = num_cols;
    resDesc.res.pitch2D.height = num_rows;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<mt>();
    resDesc.res.pitch2D.pitchInBytes = num_cols*sizeof(mt);
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    dim3 threads(4, 4);
    kernel<<<1, 1>>>(tex);
    cudaDeviceSynchronize();
    printf("\n");
    return 0;
}
