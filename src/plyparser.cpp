#include "plyparser.hpp"
#include <sstream>
#include <stdexcept>

PlyHeader parse_ply_header(std::ifstream& f) {
  PlyHeader h;
  std::string line;

  std::getline(f, line);
  if (line.rfind("ply", 0) != 0) throw std::runtime_error("not a ply");

  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::istringstream ss(line);
    std::string tok;
    ss >> tok;

    if (tok == "format") {
      std::string fmt;
      ss >> fmt;
      if (fmt == "ascii") throw std::runtime_error("ascii ply not handled");
    } else if (tok == "element") {
      std::string name;
      ss >> name;
      if (name == "vertex") ss >> h.num_gaussians;
    } else if (tok == "property") {
      std::string type, name;
      ss >> type >> name;
      if (type != "float") throw std::runtime_error("only supports float properties");
      h.dim++;
      h.prop_names.push_back(name);
    } else if (tok == "end_header") {
      break;
    }
  }

  if (h.num_gaussians == 0) throw std::runtime_error("zero vertex information");
  if (h.dim == 0)           throw std::runtime_error("no properties in header");

  h.body_start = f.tellg();
  return h;
}
