#include "rd_profile.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

std::string component_of(const std::string& name) {
  if (name == "x" || name == "y" || name == "z")    return "position";
  if (name == "nx" || name == "ny" || name == "nz") return "normal";
  if (name.rfind("f_dc_", 0) == 0)                  return "sh_dc";
  if (name.rfind("f_rest_", 0) == 0)                return "sh_rest";
  if (name == "opacity")                            return "opacity";
  if (name.rfind("scale", 0) == 0)                  return "scale";
  if (name.rfind("rot", 0) == 0)                    return "rotation";
  return "other";
}

template <typename T>
std::vector<std::vector<uint32_t>>
histogram_impl(std::ifstream& f, const Quantizer& quant, const PlyHeader& h,
               size_t chunk_size) {
  const int dim = h.dim;
  std::vector<float> chunk_buffer(chunk_size * dim);
  std::vector<T>     encoded_buffer(chunk_size * dim);
  std::vector<std::vector<uint32_t>> histograms(
      dim, std::vector<uint32_t>(quant.max_val + 1, 0));

  f.clear();
  f.seekg(h.body_start);

  size_t left = h.num_gaussians;
  while (left > 0) {
    size_t to_read = std::min(chunk_size, left);
    f.read(reinterpret_cast<char*>(chunk_buffer.data()), to_read * h.stride());
    if (!f) throw std::runtime_error("truncated body during histogram pass");

    quant.encode_chunk<T>(chunk_buffer.data(), to_read, encoded_buffer.data());

    for (size_t i = 0; i < to_read; ++i)
      for (int d = 0; d < dim; ++d)
        histograms[d][encoded_buffer[i * dim + d]]++;

    left -= to_read;
  }
  return histograms;
}

}

void accumulate_bounds(std::ifstream& f, Quantizer& quant,
                       const PlyHeader& h, size_t chunk_size) {
  std::vector<float> chunk_buffer(chunk_size * h.dim);

  f.clear();
  f.seekg(h.body_start);

  size_t left = h.num_gaussians;
  while (left > 0) {
    size_t to_read = std::min(chunk_size, left);
    f.read(reinterpret_cast<char*>(chunk_buffer.data()), to_read * h.stride());
    if (!f) throw std::runtime_error("truncated body during bounds pass");

    quant.update_bounds(chunk_buffer.data(), to_read);
    left -= to_read;
  }
}

std::vector<std::vector<uint32_t>>
build_histogram_stream(std::ifstream& f, const Quantizer& quant,
                       const PlyHeader& h, size_t chunk_size, int bits) {
  return (bits <= 8)
             ? histogram_impl<uint8_t>(f, quant, h, chunk_size)
             : histogram_impl<uint16_t>(f, quant, h, chunk_size);
}

void report_rd_profile(const std::vector<std::vector<uint32_t>>& histograms,
                       const PlyHeader& h, int bits) {
  const int dim = h.dim;
  const double N = h.num_gaussians;

  std::vector<double> Hd(dim, 0.0);
  for (int d = 0; d < dim; ++d) {
    double H = 0.0;
    for (uint32_t c : histograms[d]) {
      if (c) {
        double p = c / N;
        H -= p * std::log2(p);
      }
    }
    Hd[d] = H;
  }

  struct Agg { int dims = 0; double bits = 0.0; };
  std::map<std::string, Agg> agg;
  for (int d = 0; d < dim; ++d) {
    Agg& a = agg[component_of(h.prop_names[d])];
    a.dims++;
    a.bits += Hd[d];
  }

  static const char* order[] = {"position", "sh_dc", "sh_rest", "opacity",
                                "scale", "rotation", "normal", "other"};

  std::cout << "\n=== R-D rate profile (b=" << bits << ", "
            << h.num_gaussians << " gaussians) ===\n";
  std::cout << std::left << std::setw(10) << "component"
            << std::right << std::setw(6) << "dims"
            << std::setw(12) << "H/symbol" << std::setw(12) << "bits/gs"
            << std::setw(10) << "raw" << std::setw(9) << "ratio" << "\n";
  std::cout << std::string(59, '-') << "\n";

  double total_H = 0.0;
  for (const char* name : order) {
    auto it = agg.find(name);
    if (it == agg.end()) continue;
    const Agg& a = it->second;
    double raw = (double)a.dims * bits;
    std::cout << std::left << std::setw(10) << name
              << std::right << std::setw(6) << a.dims
              << std::setw(12) << std::fixed << std::setprecision(3) << (a.bits / a.dims)
              << std::setw(12) << a.bits
              << std::setw(10) << raw
              << std::setw(8) << std::setprecision(1) << (raw / a.bits) << "x" << "\n";
    total_H += a.bits;
  }

  std::cout << std::string(59, '-') << "\n";
  double raw_total = (double)dim * bits;
  std::cout << std::left << std::setw(10) << "TOTAL"
            << std::right << std::setw(6) << dim
            << std::setw(12) << std::setprecision(3) << (total_H / dim)
            << std::setw(12) << total_H
            << std::setw(10) << raw_total
            << std::setw(8) << std::setprecision(1) << (raw_total / total_H) << "x" << "\n";
  std::cout << "\nbits/gaussian: " << std::setprecision(1) << total_H
            << "  (raw " << raw_total << ")   file: ~" << std::setprecision(2)
            << (total_H * N / 8.0 / 1e6) << " MB entropy floor\n";
}
