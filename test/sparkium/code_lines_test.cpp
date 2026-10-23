#include "sparkium/core/code_lines.h"

#include <gtest/gtest.h>

TEST(CodeLines, RemovesFileBomBeforeComposingShaderFragments) {
  const std::string source = "#pragma once\n// 材质\nfloat value;\n";
  const std::string encoded = "\xEF\xBB\xBF" + source;
  EXPECT_EQ(std::string(sparkium::CodeLines(encoded.c_str())), source);
  EXPECT_EQ(std::string(sparkium::CodeLines(encoded)), source);
  EXPECT_EQ(std::string(sparkium::CodeLines(std::vector<uint8_t>(encoded.begin(), encoded.end()))), source);
  grassland::VirtualFileSystem vfs;
  vfs.WriteFile("material.slang", encoded);
  sparkium::CodeLines lines("namespace Material {\n}");
  lines.InsertAfter(sparkium::CodeLines(vfs, "material.slang"), "namespace Material {");
  EXPECT_EQ(std::string(lines), "namespace Material {\n" + source + "}\n");
  EXPECT_EQ(std::string(sparkium::CodeLines(source)), source);
  EXPECT_EQ(std::string(sparkium::CodeLines("")), "");
  EXPECT_EQ(std::string(sparkium::CodeLines("\xEF\xBB\xBF")), "");
}
