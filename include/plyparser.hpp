#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct PlyHeader {
  int dim = 0;
  uint32_t num_gaussians = 0;
  std::vector<std::string> prop_names;
  std::streampos body_start;

  size_t stride() const { return (size_t)dim * sizeof(float); }
};

PlyHeader parse_ply_header(std::ifstream& f);
