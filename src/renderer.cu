#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include "plyparser.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CK(c) do{ cudaError_t e=(c); if(e){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

struct Camera {
    float3 R0, R1, R2;
    float3 t;
    float fx, fy, cx, cy;
};

struct Gaussians {
  std::vector<float3> means;
  std::vector<float3> scales;
  std::vector<float4> quats;
  std::vector<float> opacity;
};

__host__ __device__ float3 sub(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ float3 add(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ float3 scale(float3 a, float s){ return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ float  dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ float3 cross(float3 a, float3 b){
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}
__host__ __device__ float3 normalize(float3 a){ float L = sqrtf(dot(a,a)); return scale(a, 1.0f/L); }

__host__ __device__ float4 normalize4(float4 q){
  float L = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  return make_float4(q.x/L, q.y/L, q.z/L, q.w/L);
}

Gaussians load_components(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open file");
  PlyHeader header = parse_ply_header(f);
  int dim = header.dim;
  size_t n = header.num_gaussians;

  std::vector<float> buffer(n * dim);
  f.clear();
  f.seekg(header.body_start);
  f.read(reinterpret_cast<char*>(buffer.data()), n * header.stride());

  Gaussians g;
  g.means.resize(n);
  g.scales.resize(n);
  g.quats.resize(n);
  g.opacity.resize(n);

  auto col = [&](const std::string& name) -> int {
    for (int j = 0; j < dim; ++j)
      if (header.prop_names[j] == name) return j;
    throw std::runtime_error("missing property: " + name);
  };
  int sx = col("scale_0"), sy = col("scale_1"), sz = col("scale_2");
  int q0 = col("rot_0"), q1 = col("rot_1"), q2 = col("rot_2"), q3 = col("rot_3");
  int op = col("opacity");

  for (size_t i = 0; i < n; ++i) {
    const float* row = &buffer[i * dim];
    g.means[i]   = make_float3(row[0], row[1], row[2]);
    g.scales[i]  = make_float3(expf(row[sx]), expf(row[sy]), expf(row[sz]));
    g.quats[i]   = normalize4(make_float4(row[q0], row[q1], row[q2], row[q3]));
    g.opacity[i] = 1.0f / (1.0f + expf(-row[op]));
  }

  return g;
}

Camera make_camera(const std::vector<float3>& means, int W, int H) {
  size_t n = means.size();

  float3 center = make_float3(0, 0, 0);
  for (size_t i = 0; i < n; ++i) center = add(center, means[i]);
  center = scale(center, 1.0f / n);

  float radius = 0.0f;
  for (size_t i = 0; i < n; ++i)
    radius += sqrtf(dot(sub(means[i], center), sub(means[i], center)));
  radius /= n;

  float az = 0.0f;
  float3 cam_pos = add(center, scale(make_float3(cosf(az), 0.2f, sinf(az)), radius * 2.5f));

  float3 up = make_float3(0, -1, 0);
  float3 forward = normalize(sub(center, cam_pos));
  float3 right = normalize(cross(forward, up));
  float3 down = cross(forward, right);

  float fov = 60.0f * 3.14159265f / 180.0f;

  Camera cam;
  cam.R0 = right;
  cam.R1 = down;
  cam.R2 = forward;
  cam.t = make_float3(-dot(right, cam_pos), -dot(down, cam_pos), -dot(forward, cam_pos));
  cam.fx = cam.fy = 0.5f * W / tanf(fov / 2);
  cam.cx = W / 2.0f;
  cam.cy = H / 2.0f;
  return cam;
}

void save_png(const char* path, const std::vector<float3>& img, int W, int H) {
  std::vector<unsigned char> pixels(W * H * 3);
  for (int i = 0; i < W * H; ++i) {
    pixels[i*3+0] = (unsigned char)(std::clamp(img[i].x, 0.f, 1.f) * 255);
    pixels[i*3+1] = (unsigned char)(std::clamp(img[i].y, 0.f, 1.f) * 255);
    pixels[i*3+2] = (unsigned char)(std::clamp(img[i].z, 0.f, 1.f) * 255);
  }
  stbi_write_png(path, W, H, 3, pixels.data(), W * 3);
}

__global__ void project(const float3* means, int n, Camera cam, float3* img, int W, int H) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 p = means[i];
  float X = dot(cam.R0, p) + cam.t.x;
  float Y = dot(cam.R1, p) + cam.t.y;
  float Z = dot(cam.R2, p) + cam.t.z;
  if (Z <= 0.0f) return;

  int u = (int)(cam.fx * X / Z + cam.cx);
  int v = (int)(cam.fy * Y / Z + cam.cy);
  if (u < 0 || u >= W || v < 0 || v >= H) return;

  img[v * W + u] = make_float3(1, 1, 1);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <file.ply>\n";
    return 1;
  }
  const int W = 800;
  const int H = 800;

  Gaussians g = load_components(argv[1]);
  Camera cam = make_camera(g.means, W, H);
  int n = (int)g.means.size();

  float3* d_means;
  float3* d_scales;
  float4* d_quats;
  float*  d_opacity;
  CK(cudaMalloc(&d_means,   n * sizeof(float3)));
  CK(cudaMalloc(&d_scales,  n * sizeof(float3)));
  CK(cudaMalloc(&d_quats,   n * sizeof(float4)));
  CK(cudaMalloc(&d_opacity, n * sizeof(float)));
  CK(cudaMemcpy(d_means,   g.means.data(),   n * sizeof(float3), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_scales,  g.scales.data(),  n * sizeof(float3), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_quats,   g.quats.data(),   n * sizeof(float4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_opacity, g.opacity.data(), n * sizeof(float),  cudaMemcpyHostToDevice));

  float3* d_img;
  CK(cudaMalloc(&d_img, W * H * sizeof(float3)));
  CK(cudaMemset(d_img, 0, W * H * sizeof(float3)));

  int TPB = 256;
  project<<<(n + TPB - 1) / TPB, TPB>>>(d_means, n, cam, d_img, W, H);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float3> h_img(W * H);
  CK(cudaMemcpy(h_img.data(), d_img, W * H * sizeof(float3), cudaMemcpyDeviceToHost));

  save_png("stage1.png", h_img, W, H);
  cudaFree(d_means);
  cudaFree(d_scales);
  cudaFree(d_quats);
  cudaFree(d_opacity);
  cudaFree(d_img);
}
