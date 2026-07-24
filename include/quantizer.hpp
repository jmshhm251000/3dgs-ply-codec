#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

struct Quantizer {
  int dim{};
  uint32_t max_val;
  std::vector<float> qMin, qMax;
  std::vector<float> scale;

  Quantizer(int dim_, int b);

  void update_bounds(const float* data, size_t n);
  void compute_scales();

  template <typename T>
  void encode(const float* data, T* out) const{
    for (int d = 0; d < dim; ++d) {
      if (scale[d] == 0.0f) {out[d] = 0; continue;}
      float v = data[d];
      if (!std::isfinite(v)) {out[d] = 0; continue;}
      int q = std::lround((v - qMin[d]) * scale[d]);
      q = std::clamp(q, 0, (int)max_val);
      out[d] = static_cast<T>(q);
    }
  }

  template <typename T>
  void encode_chunk(const float* data, size_t n, T* out) const{
    for (size_t i = 0; i < n; ++i) {
      encode<T>(data + i * dim, out + i * dim);
    }
  }

  template <typename T>
  void decode(const T* data, float* out) const{
    for (int d = 0; d < dim; ++d) {
      if (scale[d] == 0) {out[d] = qMin[d]; continue;}
      float range = qMax[d] - qMin[d];
      out[d] = qMin[d] + ((float)data[d]/(float) max_val) * range;
    }
  }
};
