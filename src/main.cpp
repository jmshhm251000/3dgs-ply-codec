#include "plyparser.hpp"
#include "quantizer.hpp"
#include "rd_profile.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <file.ply> <chunk_size> <bits>\n";
    return 1;
  }

  const size_t CHUNK_SIZE = std::stoull(argv[2]);
  const int bits = std::stoi(argv[3]);
  if (CHUNK_SIZE == 0)         throw std::runtime_error("chunk_size must be > 0");
  if (bits < 1 || bits > 16)   throw std::runtime_error("bits must be in [1, 16]");

  std::ifstream f(argv[1], std::ios::binary);
  if (!f) throw std::runtime_error("cannot open file");

  PlyHeader header = parse_ply_header(f);

  Quantizer quant(header.dim, bits);
  accumulate_bounds(f, quant, header, CHUNK_SIZE);
  quant.compute_scales();

  auto histograms = build_histogram_stream(f, quant, header, CHUNK_SIZE, bits);
  report_rd_profile(histograms, header, bits);

  return 0;
}
