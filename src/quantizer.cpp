#include "quantizer.hpp"
#include <cfloat>
#include <immintrin.h>

Quantizer::Quantizer(int dim_, int b) {
  dim = dim_;
  max_val = (1ULL << b) - 1;
  qMin.assign(dim, FLT_MAX);
  qMax.assign(dim, -FLT_MAX);

}

void Quantizer::update_bounds(const float* data, size_t n) {
  if (n == 0) return;

  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < dim; ++d) {
      float v = *(data + i * dim + d);
      if (!std::isfinite(v)) continue;
      qMin[d] = std::min(qMin[d], v);
      qMax[d] = std::max(qMax[d], v);
    }
  }
}

void Quantizer::compute_scales() {
  scale.assign(dim, 0.0f);
  for (int d = 0; d < dim; ++d) {
    float range = (qMax[d] - qMin[d]);
    scale[d] = (range > 1e-6f) ? ((float)max_val / range) : 0.0f;
  }
}
