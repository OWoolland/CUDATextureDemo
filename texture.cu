#include <stdio.h>
#include <stdint.h>
#include <iostream>

//typedef uint8_t float;  // use an integer type

// -----------------------------------------------------------------------------
// Helpers

void defineInput (float* dataIn, int num_rows, int num_cols) {
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

  return;
}

void allocate2DMemory (float*& ptr_d,
                       size_t& pitch,
                       int num_rows,
                       int num_cols,
                       void* dataIn
                       ) {

  cudaMallocPitch((void**)&ptr_d,
                  &pitch,
                  num_cols*sizeof(float),
                  num_rows);
  
  cudaMemcpy2D(ptr_d,
               pitch,
               dataIn,
               num_cols*sizeof(float),
               num_cols*sizeof(float),
               num_rows,
               cudaMemcpyHostToDevice);

  return;
}

void configureResourceDescription (struct cudaResourceDesc& resDesc,
                                   float* ptr_d,
                                   size_t& pitch,
                                   int num_rows,
                                   int num_cols
                                   ) {
  memset(&resDesc, 0, sizeof(resDesc));
    
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = ptr_d;
  resDesc.res.pitch2D.width = num_cols;
  resDesc.res.pitch2D.height = num_rows;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  resDesc.res.pitch2D.pitchInBytes = pitch;

  return;
}

void configureTexture (struct cudaTextureDesc& texDesc) {
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = true;

  return;
} 

// -----------------------------------------------------------------------------
// Interpolation kernel

__global__ void kernel (cudaTextureObject_t tex,
                        float xMax,
                        float yMax,
                        int num_rows,
                        int num_cols
                        )
{
  float offset = 0.5;
  float xScale = 1. / xMax;
  float yScale = 1. / xMax;

  printf("Output\n");
  for (int ii = 0; ii < num_rows; ++ii) {
    for (int jj = 0; jj < num_cols; ++jj) {
      float x = (float)jj;
      float y = (float)ii;
      float val = tex2D<float>(tex, (x+offset)*xScale, (y+offset)*yScale);
      printf("%.2f, ", val);
    }
    printf("\n");
  }
}

// -----------------------------------------------------------------------------
// Main

int main(int argc, char **argv)
{
  const int num_rows = 4;
  const int num_cols = 4;

  // ---------------------------------------------------------------------------
  // Get input data
  
  float dataIn[num_cols*num_rows*sizeof(float)];
  defineInput(dataIn, num_rows, num_cols);

  // ---------------------------------------------------------------------------
  // Allocate pitched (2D) memory
    
  float* ptr_d = 0;
  size_t pitch = 0;
  allocate2DMemory(ptr_d, pitch, num_rows, num_cols, dataIn);

  // ---------------------------------------------------------------------------
  // Initialise pitched memory
    
  struct cudaResourceDesc resDesc;
  configureResourceDescription(resDesc, ptr_d, pitch, num_rows, num_cols);

  // ---------------------------------------------------------------------------
  // Configure interpolation parameters
    
  struct cudaTextureDesc texDesc;
  configureTexture(texDesc);

  // ---------------------------------------------------------------------------
  // Create texture

  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  // ---------------------------------------------------------------------------
  // Fetch interpolated parameters

  float xMax = num_cols;
  float yMax = num_rows;

  dim3 threads(1, 1);
  kernel<<<1, threads>>>(tex, xMax, yMax, num_rows, num_cols);

  // ---------------------------------------------------------------------------
  // Await completion and exit

  cudaDeviceSynchronize();
  return 0;
}

