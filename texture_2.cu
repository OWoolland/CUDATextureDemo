#include <stdio.h>
#include <iostream>

__global__ void fetch(cudaTextureObject_t tex, std::size_t width, std::size_t height) 
{
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      float u = i;//(i + 0.5f) / width;
      float v = j;//(j + 0.5f) / height;

      auto p = tex2D<float4>(tex, u, v);
      printf("i=%d, j=%d -> u=%3.2f, v=%3.2f, r=%d, g=%d, b=%d, a=%d\n", i, j, u, v, p.x, p.y, p.z, p.w);
      // -> always returns p = {0, 0, 0, 0}
    }
  }
}

int main() {

  constexpr std::size_t width = 4;
  constexpr std::size_t height = 4;

  // creating a dummy texture
  float4 image[width*height];
  for(std::size_t j = 0; j < height; ++j) {
    for(std::size_t i = 0; i < width; ++i) {
      float4 foo = make_float4(j*width+i, 1, 1, 1);
      image[j*width+i] = foo;
      std::cout << "("
                << foo.x << ","
                << foo.y << ","
                << foo.z << ","
                << foo.w << ")"
                << " ";
    }
    std::cout << std::endl;
  }

  cudaArray_t cuArray;
  auto channelDesc = cudaCreateChannelDesc<float4>();
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpy2DToArray(cuArray, 0, 0, image, width*sizeof(float4), width*sizeof(float4), height, cudaMemcpyHostToDevice);

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
  texDesc.normalizedCoords = false;
  texDesc.normalized       = false;

  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  fetch<<<1, 1>>>(texObj, width, height);

  cudaDeviceSynchronize();

  cudaDestroyTextureObject(texObj);
  cudaFreeArray(cuArray);

  return 0;
}
