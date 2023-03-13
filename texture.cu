#include <stdio.h>
#include <iostream>

__global__ void fetch(cudaTextureObject_t tex, std::size_t width, std::size_t height) 
{
  printf("\nWidth: %d, Height: %d\n", width, height);

  for (int ii = 0 ; ii < height ; ++ii) {
    for (int jj = 0 ; jj < width ; ++jj) {
      float base = (jj*width+ii);
      float u = base+0.5;
      //float v = base+0.5;
      auto a = tex1Dfetch<float>(tex, u);
      //auto p = tex2Dfetch<float>(tex, u, v);
      printf("u = %f\n", u);
      //printf("v = %f\n", v);
      printf("ax = %f\n", a);
      //float ay = a.y;
      //float az = a.z;
      //printf("px %f, py %f, pz %f", px, py, pz);
      //printf("u=%d v=%d\n\t ii %d width %d\n", u, v, ii, width);
    }
  }

  for (int j = 0; j < height; j++) {
    //printf("j=%d\n",j);
    for (int i = 0; i < width; i++) {


      //auto p = tex2D<float>(tex, u, v);
      //printf("i=%d, j=%d -> u=%f, v=%f, r=%d, g=%f, b=%f, a=%f\n", i, j, u, v, p.x, p.y, p.z, p.w);
      //      printf("i=%d, j=%d\n",i,j);
      //printf("i=%d, j=%d, px=%f, py=%f, pz=%f,");//,i,j,p.x,p.y,p.z);
      // -> always returns p = {0, 0, 0, 0}
    }
  }
}

int main() {

  constexpr std::size_t width = 10;
  constexpr std::size_t height = 10;

  // creating a dummy texture
  float image[width*height];
  for(std::size_t jj = 0; jj < height; ++jj) {
    for(std::size_t ii = 0; ii < width; ++ii) {
      image[jj*width+ii] = (jj*width+ii);
      std::cout << image[jj*width+ii] << " " ;
    }
    std::cout << std::endl;
  }

  cudaArray_t cuArray;
  auto channelDesc = cudaCreateChannelDesc<float>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpy2DToArray(cuArray, 0, 0, image, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeBorder;
  texDesc.addressMode[1]   = cudaAddressModeBorder;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  fetch<<<1, 1>>>(texObj, width, height);

  cudaDeviceSynchronize();

  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);

  return 0;
}
