#pragma once

#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

struct CellsPattern {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> cells;
};

inline CellsPattern ParseCellsPattern(std::istream &input) {
  CellsPattern pattern;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty() || line.front() == '!')
      continue;
    if (pattern.width == 0)
      pattern.width = static_cast<int>(line.size());
    if (static_cast<int>(line.size()) != pattern.width)
      throw std::invalid_argument("Pattern rows must have equal width");
    for (char cell : line) {
      if (cell != '.' && cell != 'O')
        throw std::invalid_argument("Pattern cells must be '.' or 'O'");
      pattern.cells.push_back(cell == 'O');
    }
    ++pattern.height;
  }
  if (pattern.height == 0)
    throw std::invalid_argument("Pattern is empty");
  return pattern;
}

inline CellsPattern LoadCellsPattern(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Cannot open pattern file: " + path);
  return ParseCellsPattern(input);
}

inline std::vector<uint8_t> CenterCellsPattern(const CellsPattern &pattern, int width, int height) {
  if (pattern.width <= 0 || pattern.height <= 0 || pattern.width > width || pattern.height > height ||
      pattern.cells.size() != static_cast<size_t>(pattern.width * pattern.height))
    throw std::invalid_argument("Pattern does not fit the cell grid");

  std::vector<uint8_t> grid(static_cast<size_t>(width) * height, 0);
  const int left = (width - pattern.width) / 2;
  const int top = (height - pattern.height) / 2;
  for (int y = 0; y < pattern.height; ++y)
    for (int x = 0; x < pattern.width; ++x)
      grid[(top + y) * width + left + x] = pattern.cells[y * pattern.width + x];
  return grid;
}
