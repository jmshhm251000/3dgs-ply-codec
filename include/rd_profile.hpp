#pragma once
#include "plyparser.hpp"
#include "quantizer.hpp"
#include <cstdint>
#include <fstream>
#include <vector>

void accumulate_bounds(std::ifstream& f, Quantizer& quant,
                       const PlyHeader& h, size_t chunk_size);

std::vector<std::vector<uint32_t>>
build_histogram_stream(std::ifstream& f, const Quantizer& quant,
                       const PlyHeader& h, size_t chunk_size, int bits);

void report_rd_profile(const std::vector<std::vector<uint32_t>>& histograms,
                       const PlyHeader& h, int bits);
