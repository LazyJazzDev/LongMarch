#include <gtest/gtest.h>

#include <chrono>
#include <sstream>

#include "../../demo/gol/cells_pattern.h"

TEST(CellsFile, RoundTripPreservesDimensionsBordersAndEveryCell) {
  CellsPattern original{7, 5, std::vector<uint8_t>(35, 0)};
  original.cells[9] = original.cells[17] = original.cells[25] = 1;
  std::stringstream file;
  WriteCellsPattern(file, original);
  const auto restored = ParseCellsPattern(file);
  EXPECT_EQ(restored.width, original.width);
  EXPECT_EQ(restored.height, original.height);
  EXPECT_EQ(restored.cells, original.cells);
}

TEST(CellsFile, RoundTripPreservesAnEmptyMaximumGrid) {
  CellsPattern original{200, 200, std::vector<uint8_t>(40000, 0)};
  std::stringstream file;
  WriteCellsPattern(file, original);
  const auto restored = ParseCellsPattern(file);
  EXPECT_EQ(restored.width, 200);
  EXPECT_EQ(restored.height, 200);
  EXPECT_EQ(restored.cells, original.cells);
}

TEST(CellsFile, AcceptsCommentsCRLFAndSingleCellPatterns) {
  std::istringstream input("!Name: test\r\n\r\nO\r\n");
  auto pattern = ParseCellsPattern(input);
  EXPECT_EQ(pattern.width, 1);
  EXPECT_EQ(pattern.height, 1);
  EXPECT_EQ(CenterCellsPattern(pattern, 2, 2), (std::vector<uint8_t>{1, 0, 0, 0}));
}

TEST(CellsFile, RejectsMalformedAndOversizedPatterns) {
  for (const auto &data :
       {std::string("! empty\n"), std::string("OO\n.\n"), std::string("OX\n"), std::string(201, 'O')}) {
    std::istringstream file(data);
    EXPECT_THROW(ParseCellsPattern(file), std::invalid_argument);
  }
  std::string tall;
  for (int i = 0; i < 201; ++i)
    tall += "O\n";
  std::istringstream file(tall);
  EXPECT_THROW(ParseCellsPattern(file), std::invalid_argument);
}

TEST(CellsFile, RejectsInvalidGridBeforeWriting) {
  for (const auto &pattern : {CellsPattern{2, 2, {0, 1}}, CellsPattern{0, 2, {}}, CellsPattern{2, 2, {0, 2, 0, 0}}}) {
    std::ostringstream file;
    EXPECT_THROW(WriteCellsPattern(file, pattern), std::invalid_argument);
    EXPECT_TRUE(file.str().empty());
  }
}

TEST(CellsFile, ReportsStreamFailures) {
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EXPECT_THROW(WriteCellsPattern(output, {2, 2, {0, 1, 1, 0}}), std::runtime_error);
  std::istringstream input("OO\nOO\n");
  input.setstate(std::ios::badbit);
  EXPECT_THROW(ParseCellsPattern(input), std::runtime_error);
}

TEST(CellsFile, SavesAndLoadsUnicodeFilePaths) {
  auto path = std::filesystem::temp_directory_path() /
              std::filesystem::u8path(
                  "gol-局面-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".cells");
  const CellsPattern original{3, 2, {1, 0, 0, 0, 0, 1}};
  SaveCellsPattern(path.u8string(), original);
  const auto restored = LoadCellsPattern(path.u8string());
  EXPECT_EQ(restored.width, 3);
  EXPECT_EQ(restored.height, 2);
  EXPECT_EQ(restored.cells, original.cells);
  // Validation occurs before opening an existing file for replacement.
  EXPECT_THROW(SaveCellsPattern(path.u8string(), {0, 0, {}}), std::invalid_argument);
  EXPECT_EQ(LoadCellsPattern(path.u8string()).cells, original.cells);
  std::filesystem::remove(path);
  EXPECT_THROW(LoadCellsPattern(path.u8string()), std::runtime_error);
}

TEST(CellsFile, FitPreservesLargerDimensionsAndCentersEachAxis) {
  const CellsPattern pattern{3, 2, {1, 0, 1, 0, 1, 0}};
  for (auto dimensions : {std::pair{8, 7}, std::pair{8, 2}, std::pair{2, 7}, std::pair{2, 2}}) {
    const auto fitted = FitCellsPattern(pattern, dimensions.first, dimensions.second);
    const int width = std::max(3, dimensions.first);
    const int height = std::max(2, dimensions.second);
    EXPECT_EQ(fitted.width, width);
    EXPECT_EQ(fitted.height, height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int px = x - (width - 3) / 2;
        const int py = y - (height - 2) / 2;
        const int expected = px >= 0 && px < 3 && py >= 0 && py < 2 ? pattern.cells[py * 3 + px] : 0;
        EXPECT_EQ(fitted.cells[y * width + x], expected);
      }
    }
  }
}

TEST(CellsFile, FitPadsSingleCellAndClearsSurroundingGrid) {
  auto fitted = FitCellsPattern({1, 1, {1}}, 200, 200);
  EXPECT_EQ(fitted.width, 200);
  EXPECT_EQ(fitted.height, 200);
  EXPECT_EQ(std::count(fitted.cells.begin(), fitted.cells.end(), 1), 1);
  EXPECT_EQ(fitted.cells[99 * 200 + 99], 1);
}
