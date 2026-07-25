#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CK(c) do{ cudaError_t e=(c); if(e){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__global__ void paint(float3* img, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    img[y * W + x] = make_float3((float)x / W, (float)y / H, 0.2f);
}

int main() {
  const int W = 800;
  const int H = 800;

  float3* d_img;
  CK(cudaMalloc(&d_img, W * H * sizeof(float3)));

  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16);
  paint<<<grid, block>>>(d_img, W, H);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float3> h_img(W * H);
  CK(cudaMemcpy(h_img.data(), d_img, W * H * sizeof(float3), cudaMemcpyDeviceToHost));

  std::vector<unsigned char> pixels(W * H * 3);
  for (int i = 0; i < W * H; ++i) {
      pixels[i*3+0] = (unsigned char)(std::clamp(h_img[i].x, 0.f, 1.f) * 255);
      pixels[i*3+1] = (unsigned char)(std::clamp(h_img[i].y, 0.f, 1.f) * 255);
      pixels[i*3+2] = (unsigned char)(std::clamp(h_img[i].z, 0.f, 1.f) * 255);
  }
  stbi_write_png("stage0.png", W, H, 3, pixels.data(), W * 3);

  cudaFree(d_img);
}
