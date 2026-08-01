#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
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
  std::vector<float3> colors;
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

__device__ void compute_cov3d(float4 q, float3 s, float cov[6]) {
  float w = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  float3 rc0 = make_float3(1 - 2 * (y*y + z*z),  2 * (x*y + w*z),  2 * (x*z - w*y));
  float3 rc1 = make_float3(2 * (x*y - w*z),  1 - 2 * (x*x + z*z),  2 * (y*z + w*x));
  float3 rc2 = make_float3(2 * (x*z + w*y),  2 * (y*z - w*x),  1 - 2 * (x*x + y*y));

  float3 m0 = scale(rc0, s.x);
  float3 m1 = scale(rc1, s.y);
  float3 m2 = scale(rc2, s.z);

  cov[0] = m0.x*m0.x + m1.x*m1.x + m2.x*m2.x;
  cov[1] = m0.x*m0.y + m1.x*m1.y + m2.x*m2.y;
  cov[2] = m0.x*m0.z + m1.x*m1.z + m2.x*m2.z;
  cov[3] = m0.y*m0.y + m1.y*m1.y + m2.y*m2.y;
  cov[4] = m0.y*m0.z + m1.y*m1.z + m2.y*m2.z;
  cov[5] = m0.z*m0.z + m1.z*m1.z + m2.z*m2.z;
}

__device__ float3 cov3_mul(const float cov[6], float3 v) {
  return make_float3(cov[0]*v.x + cov[1]*v.y + cov[2]*v.z,
                     cov[1]*v.x + cov[3]*v.y + cov[4]*v.z,
                     cov[2]*v.x + cov[4]*v.y + cov[5]*v.z);
}

__device__ float3 compute_cov2d(float3 tc, const float cov[6], const Camera& cam) {
  float z2 = tc.z * tc.z;

  float3 j0 = make_float3(cam.fx / tc.z, 0.0f, -cam.fx * tc.x / z2);
  float3 j1 = make_float3(0.0f, cam.fy / tc.z, -cam.fy * tc.y / z2);

  float3 m0 = add(add(scale(cam.R0, j0.x), scale(cam.R1, j0.y)), scale(cam.R2, j0.z));
  float3 m1 = add(add(scale(cam.R0, j1.x), scale(cam.R1, j1.y)), scale(cam.R2, j1.z));

  float a = dot(m0, cov3_mul(cov, m0)) + 0.3f;
  float b = dot(m0, cov3_mul(cov, m1));
  float c = dot(m1, cov3_mul(cov, m1)) + 0.3f;

  return make_float3(a, b, c);
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
  g.colors.resize(n);

  auto col = [&](const std::string& name) -> int {
    for (int j = 0; j < dim; ++j)
      if (header.prop_names[j] == name) return j;
    throw std::runtime_error("missing property: " + name);
  };
  int sx = col("scale_0"), sy = col("scale_1"), sz = col("scale_2");
  int q0 = col("rot_0"), q1 = col("rot_1"), q2 = col("rot_2"), q3 = col("rot_3");
  int op = col("opacity");
  int f0 = col("f_dc_0"), f1 = col("f_dc_1"), f2 = col("f_dc_2");

  const float SH_C0 = 0.2820947917738781f;

  for (size_t i = 0; i < n; ++i) {
    const float* row = &buffer[i * dim];
    g.means[i]   = make_float3(row[0], row[1], row[2]);
    g.scales[i]  = make_float3(expf(row[sx]), expf(row[sy]), expf(row[sz]));
    g.quats[i]   = normalize4(make_float4(row[q0], row[q1], row[q2], row[q3]));
    g.opacity[i] = 1.0f / (1.0f + expf(-row[op]));
    g.colors[i]  = make_float3(0.5f + SH_C0 * row[f0],
                               0.5f + SH_C0 * row[f1],
                               0.5f + SH_C0 * row[f2]);
  }

  return g;
}

Camera make_camera(const std::vector<float3>& means, int W, int H, float az) {
  size_t n = means.size();

  float3 center = make_float3(0, 0, 0);
  for (size_t i = 0; i < n; ++i) center = add(center, means[i]);
  center = scale(center, 1.0f / n);

  float radius = 0.0f;
  for (size_t i = 0; i < n; ++i)
    radius += sqrtf(dot(sub(means[i], center), sub(means[i], center)));
  radius /= n;

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

#define TILE 16

struct Splat {
  float2 xy;
  float3 conic;
  float3 color;
  float depth;
  float opacity;
  int4 rect;
};

__global__ void preprocess(const float3* means, const float3* scales, const float4* quats,
                           const float* opacity, const float3* colors, int n, Camera cam,
                           int W, int H, int tiles_x, int tiles_y,
                           Splat* splats, int* touched) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  touched[i] = 0;

  float cov[6];
  compute_cov3d(quats[i], scales[i], cov);

  float3 p = means[i];
  float X = dot(cam.R0, p) + cam.t.x;
  float Y = dot(cam.R1, p) + cam.t.y;
  float Z = dot(cam.R2, p) + cam.t.z;
  if (Z <= 0.0f) return;

  float3 cov2d = compute_cov2d(make_float3(X, Y, Z), cov, cam);
  float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
  if (det == 0.0f) return;
  float3 conic = make_float3(cov2d.z / det, -cov2d.y / det, cov2d.x / det);

  float mid = 0.5f * (cov2d.x + cov2d.z);
  float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
  int radius = (int)ceilf(3.0f * sqrtf(lambda1));

  float cu = cam.fx * X / Z + cam.cx;
  float cv = cam.fy * Y / Z + cam.cy;

  int rminx = min(tiles_x, max(0, (int)((cu - radius) / TILE)));
  int rminy = min(tiles_y, max(0, (int)((cv - radius) / TILE)));
  int rmaxx = min(tiles_x, max(0, (int)((cu + radius + TILE - 1) / TILE)));
  int rmaxy = min(tiles_y, max(0, (int)((cv + radius + TILE - 1) / TILE)));
  int cnt = (rmaxx - rminx) * (rmaxy - rminy);
  if (cnt == 0) return;

  Splat s;
  s.xy = make_float2(cu, cv);
  s.conic = conic;
  s.color = colors[i];
  s.depth = Z;
  s.opacity = opacity[i];
  s.rect = make_int4(rminx, rminy, rmaxx, rmaxy);
  splats[i] = s;
  touched[i] = cnt;
}

__global__ void duplicate(const Splat* splats, const int* offsets, const int* touched,
                          int n, int tiles_x, uint64_t* keys, int* vals) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || touched[i] == 0) return;

  int off = offsets[i];
  Splat s = splats[i];
  uint32_t dbits = __float_as_uint(s.depth);

  for (int ty = s.rect.y; ty < s.rect.w; ty++)
    for (int tx = s.rect.x; tx < s.rect.z; tx++) {
      uint64_t tile = (uint64_t)(ty * tiles_x + tx);
      keys[off] = (tile << 32) | dbits;
      vals[off] = i;
      off++;
    }
}

__global__ void find_ranges(const uint64_t* keys, int L, int2* ranges) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= L) return;

  uint32_t tile = (uint32_t)(keys[l] >> 32);
  if (l == 0) {
    ranges[tile].x = 0;
  } else {
    uint32_t prev = (uint32_t)(keys[l - 1] >> 32);
    if (tile != prev) {
      ranges[prev].y = l;
      ranges[tile].x = l;
    }
  }
  if (l == L - 1) ranges[tile].y = L;
}

__global__ void render(const Splat* splats, const int* vals, const int2* ranges,
                       int tiles_x, int W, int H, float3* img) {
  int px = blockIdx.x * TILE + threadIdx.x;
  int py = blockIdx.y * TILE + threadIdx.y;
  int tile = blockIdx.y * tiles_x + blockIdx.x;
  bool inside = (px < W && py < H);

  int2 rng = ranges[tile];
  float T = 1.0f;
  float3 C = make_float3(0.0f, 0.0f, 0.0f);

  for (int l = rng.x; l < rng.y && inside; l++) {
    Splat s = splats[vals[l]];
    float dx = px - s.xy.x;
    float dy = py - s.xy.y;
    float power = -0.5f * (s.conic.x*dx*dx + 2.0f*s.conic.y*dx*dy + s.conic.z*dy*dy);
    if (power > 0.0f) continue;

    float alpha = fminf(0.99f, s.opacity * expf(power));
    if (alpha < 1.0f / 255.0f) continue;

    C.x += T * alpha * s.color.x;
    C.y += T * alpha * s.color.y;
    C.z += T * alpha * s.color.z;
    T *= (1.0f - alpha);
    if (T < 1e-4f) break;
  }

  if (inside) img[py * W + px] = C;
}

std::vector<float3> render_view(const Camera& cam, int W, int H, int n,
                                const float3* d_means, const float3* d_scales,
                                const float4* d_quats, const float* d_opacity,
                                const float3* d_colors) {
  int tiles_x = (W + TILE - 1) / TILE;
  int tiles_y = (H + TILE - 1) / TILE;
  int num_tiles = tiles_x * tiles_y;

  Splat* d_splats;
  int* d_touched;
  int* d_offsets;
  CK(cudaMalloc(&d_splats,  n * sizeof(Splat)));
  CK(cudaMalloc(&d_touched, n * sizeof(int)));
  CK(cudaMalloc(&d_offsets, n * sizeof(int)));

  int TPB = 256;
  int blocks = (n + TPB - 1) / TPB;
  preprocess<<<blocks, TPB>>>(d_means, d_scales, d_quats, d_opacity, d_colors,
                              n, cam, W, H, tiles_x, tiles_y, d_splats, d_touched);
  CK(cudaGetLastError());

  thrust::device_ptr<int> t_touched(d_touched);
  thrust::device_ptr<int> t_offsets(d_offsets);
  thrust::exclusive_scan(t_touched, t_touched + n, t_offsets);

  int last_off = 0, last_cnt = 0;
  CK(cudaMemcpy(&last_off, d_offsets + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&last_cnt, d_touched + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
  int L = last_off + last_cnt;

  uint64_t* d_keys;
  int* d_vals;
  CK(cudaMalloc(&d_keys, L * sizeof(uint64_t)));
  CK(cudaMalloc(&d_vals, L * sizeof(int)));
  duplicate<<<blocks, TPB>>>(d_splats, d_offsets, d_touched, n, tiles_x, d_keys, d_vals);
  CK(cudaGetLastError());

  thrust::device_ptr<uint64_t> t_keys(d_keys);
  thrust::device_ptr<int> t_vals(d_vals);
  thrust::sort_by_key(t_keys, t_keys + L, t_vals);

  int2* d_ranges;
  CK(cudaMalloc(&d_ranges, num_tiles * sizeof(int2)));
  CK(cudaMemset(d_ranges, 0, num_tiles * sizeof(int2)));
  find_ranges<<<(L + TPB - 1) / TPB, TPB>>>(d_keys, L, d_ranges);
  CK(cudaGetLastError());

  float3* d_img;
  CK(cudaMalloc(&d_img, W * H * sizeof(float3)));
  CK(cudaMemset(d_img, 0, W * H * sizeof(float3)));

  dim3 grid(tiles_x, tiles_y);
  dim3 block(TILE, TILE);
  render<<<grid, block>>>(d_splats, d_vals, d_ranges, tiles_x, W, H, d_img);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float3> h_img(W * H);
  CK(cudaMemcpy(h_img.data(), d_img, W * H * sizeof(float3), cudaMemcpyDeviceToHost));

  cudaFree(d_splats);
  cudaFree(d_touched);
  cudaFree(d_offsets);
  cudaFree(d_keys);
  cudaFree(d_vals);
  cudaFree(d_ranges);
  cudaFree(d_img);
  return h_img;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <file.ply>\n";
    return 1;
  }
  const int W = 800;
  const int H = 800;

  Gaussians g = load_components(argv[1]);
  int n = (int)g.means.size();

  float3* d_means;
  float3* d_scales;
  float4* d_quats;
  float*  d_opacity;
  float3* d_colors;
  CK(cudaMalloc(&d_means,   n * sizeof(float3)));
  CK(cudaMalloc(&d_scales,  n * sizeof(float3)));
  CK(cudaMalloc(&d_quats,   n * sizeof(float4)));
  CK(cudaMalloc(&d_opacity, n * sizeof(float)));
  CK(cudaMalloc(&d_colors,  n * sizeof(float3)));
  CK(cudaMemcpy(d_means,   g.means.data(),   n * sizeof(float3), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_scales,  g.scales.data(),  n * sizeof(float3), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_quats,   g.quats.data(),   n * sizeof(float4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_opacity, g.opacity.data(), n * sizeof(float),  cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_colors,  g.colors.data(),  n * sizeof(float3), cudaMemcpyHostToDevice));

  const int VIEWS = 8;
  const float PI = 3.14159265f;
  for (int v = 0; v < VIEWS; ++v) {
    float az = v * (2.0f * PI / VIEWS);
    Camera cam = make_camera(g.means, W, H, az);
    std::vector<float3> img =
        render_view(cam, W, H, n, d_means, d_scales, d_quats, d_opacity, d_colors);

    char path[64];
    snprintf(path, sizeof(path), "orbit_%02d.png", v);
    save_png(path, img, W, H);
    printf("wrote %s  (az=%.0f deg)\n", path, az * 180.0f / PI);
  }

  cudaFree(d_means);
  cudaFree(d_scales);
  cudaFree(d_quats);
  cudaFree(d_opacity);
  cudaFree(d_colors);
}
