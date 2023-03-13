#include <stdio.h>
#include <stdint.h>
#include <iostream>

//typedef uint8_t float;  // use an integer type

const int num_rows = 4;
const int num_cols = 4;

__global__ void kernel(cudaTextureObject_t tex, float xMax, float yMax)
{
  float offset = 0.5;
  float xScale = 1. / xMax;
  float yScale = 1. / xMax;

  printf("No offset \n");
  for (int ii = 0; ii < num_rows; ++ii) {
    for (int jj = 0; jj < num_cols; ++jj) {
      float x = (float)jj;
      float y = (float)ii;
      float val = tex2D<float>(tex, (x+offset)*xScale, (y+offset)*yScale);
      printf("%.2f, ", val);
    }
    printf("\n");
  }
  printf("\n");
  
  printf("X offset (half bin)\n");
  for (int ii = 0; ii < num_rows; ++ii) {
    for (int jj = 0; jj < num_cols; ++jj) {
      float x = (float)jj+0.5;
      float y = (float)ii;
      float val = tex2D<float>(tex, (x+offset)*xScale, (y+offset)*yScale);
      printf("%.2f, ", val);
    }
    printf("\n");
  }
  printf("\n");

  printf("Y offset (half bin)\n");
  for (int ii = 0; ii < num_rows; ++ii) {
    for (int jj = 0; jj < num_cols; ++jj) {
      float x = (float)jj;
      float y = (float)ii+0.5;
      float val = tex2D<float>(tex, (x+offset)*xScale, (y+offset)*yScale);
      printf("%.2f, ", val);
    }
    printf("\n");
  }
  printf("\n");

  printf("X and Y offset (half bin)\n");
  for (int ii = 0; ii < num_rows; ++ii) {
    for (int jj = 0; jj < num_cols; ++jj) {
      float x = (float)jj+0.5;
      float y = (float)ii+0.5;
      float val = tex2D<float>(tex, (x+offset)*xScale, (y+offset)*yScale);
      printf("%.2f, ", val);
    }
    printf("\n");
  }
  printf("\n");
}

//const int num_cols = prop.texturePitchAlignment*1; // should be able to use a different multiplier here

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("texturePitchAlignment: %lu\n", prop.texturePitchAlignment);

    float xMax = num_cols;
    float yMax = num_rows;
    
    cudaTextureObject_t tex;
    float dataIn[num_cols*num_rows*sizeof(float)];
    std::cout << "Input" << std::endl;
    for (int ii = 0; ii < num_rows; ii++) {
      for (int jj = 0; jj < num_cols; jj++) {
        int index = (ii*num_cols)+jj;
        dataIn[index] = (float)index*1.5;
        std::cout << dataIn[index] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    
    float* dataDev = 0;
    size_t pitch;
    
    cudaMallocPitch((void**)&dataDev, &pitch,  num_cols*sizeof(float), num_rows);
    cudaMemcpy2D(dataDev, pitch, dataIn, num_cols*sizeof(float), num_cols*sizeof(float), num_rows, cudaMemcpyHostToDevice);
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width = num_cols;
    resDesc.res.pitch2D.height = num_rows;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.pitchInBytes = pitch;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    dim3 threads(1, 1);
    kernel<<<1, threads>>>(tex, xMax, yMax);
    cudaDeviceSynchronize();
    printf("\n");
    return 0;
}
