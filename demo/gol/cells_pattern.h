#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

#include "grid_size.h"

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
    if (line.size() > grid_size::kMax || pattern.height >= grid_size::kMax)
      throw std::invalid_argument("Pattern must fit within a 200 x 200 grid");
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
  if (input.bad())
    throw std::runtime_error("Could not read the complete pattern");
  if (pattern.height == 0)
    throw std::invalid_argument("Pattern is empty");
  return pattern;
}

inline CellsPattern LoadCellsPattern(const std::string &path) {
  const auto file = std::filesystem::u8path(path);
  if (std::filesystem::file_size(file) > 1024 * 1024)
    throw std::invalid_argument("Pattern file exceeds 1 MiB");
  std::ifstream input(file);
  if (!input)
    throw std::runtime_error("Cannot open pattern file: " + path);
  return ParseCellsPattern(input);
}

inline void ValidateCellsPattern(const CellsPattern &pattern) {
  if (pattern.width < 1 || pattern.height < 1 || pattern.width > grid_size::kMax || pattern.height > grid_size::kMax ||
      pattern.cells.size() != static_cast<size_t>(pattern.width) * pattern.height ||
      std::any_of(pattern.cells.begin(), pattern.cells.end(), [](uint8_t cell) { return cell > 1; }))
    throw std::invalid_argument("Invalid grid dimensions or cell data");
}

inline void WriteCellsPattern(std::ostream &output, const CellsPattern &pattern) {
  ValidateCellsPattern(pattern);
  output << "!Name: LongMarch saved grid\n! Full grid, including empty borders\n";
  for (int y = 0; y < pattern.height; ++y) {
    for (int x = 0; x < pattern.width; ++x)
      output.put(pattern.cells[y * pattern.width + x] ? 'O' : '.');
    output.put('\n');
  }
  if (!output)
    throw std::runtime_error("Could not write the complete grid");
}

inline void SaveCellsPattern(const std::string &path, const CellsPattern &pattern) {
  ValidateCellsPattern(pattern);
  std::ofstream output(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("Cannot save grid to: " + path);
  WriteCellsPattern(output, pattern);
  output.close();
  if (!output)
    throw std::runtime_error("Could not finish saving grid to: " + path);
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

// File loading preserves larger current dimensions and expands smaller ones.
inline CellsPattern FitCellsPattern(const CellsPattern &pattern, int current_width, int current_height) {
  ValidateCellsPattern(pattern);
  const int width = std::max(pattern.width, std::clamp(current_width, grid_size::kMin, grid_size::kMax));
  const int height = std::max(pattern.height, std::clamp(current_height, grid_size::kMin, grid_size::kMax));
  return {width, height, CenterCellsPattern(pattern, width, height)};
}
