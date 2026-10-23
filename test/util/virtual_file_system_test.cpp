#include "grassland/util/virtual_file_system.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <utility>

namespace {
using grassland::VirtualFileSystem;

std::string Read(const VirtualFileSystem &fs, const std::string &path) {
  std::vector<uint8_t> bytes;
  EXPECT_EQ(fs.ReadFile(path, bytes), 0) << path;
  return {bytes.begin(), bytes.end()};
}

TEST(VirtualFileSystem, SeparatorsAndLexicalPathsAreHostIndependent) {
  VirtualFileSystem fs;
  ASSERT_EQ(fs.WriteFile(R"(shaders\nested\..\main.slang)", "source"), 0);
  for (const auto &path : {"shaders/main.slang", "/shaders/main.slang", "///shaders//./main.slang",
                           R"(\shaders\main.slang)", R"(shaders\.\main.slang)", "missing/../shaders/main.slang"})
    EXPECT_EQ(Read(fs, path), "source");
  EXPECT_EQ(fs.WriteFile("shaders/./main.slang", "replacement"), 0);
  EXPECT_EQ(Read(fs, R"(shaders\main.slang)"), "replacement");
  ASSERT_EQ(fs.WriteFile(u8"着色器\\材质.slang", u8"内容"), 0);
  EXPECT_EQ(Read(fs, u8"着色器/材质.slang"), u8"内容");
}

TEST(VirtualFileSystem, InvalidPathsCannotMutateTheTreeOrReadOutput) {
  VirtualFileSystem fs;
  for (const auto &path : {"", "/", "\\", ".", "..", "a/", "a/.", "a/..", "../escape", "/../escape", "a/../../escape",
                           R"(a\..\..\escape)", R"(C:\file)", "C:file", "a:stream"}) {
    EXPECT_NE(fs.WriteFile(path, "bad"), 0) << path;
    std::vector<uint8_t> output{42};
    EXPECT_NE(fs.ReadFile(path, output), 0) << path;
    EXPECT_EQ(output, std::vector<uint8_t>{42});
  }
  EXPECT_NE(fs.WriteFile(std::string("a\0b", 3), "bad"), 0);
  // Failed directory-like writes must not create directories or empty files.
  EXPECT_EQ(fs.WriteFile("a", "good"), 0);
  EXPECT_EQ(fs.WriteFile("escape", "good"), 0);
}

TEST(VirtualFileSystem, FilesAndImplicitDirectoriesCannotOverlap) {
  VirtualFileSystem fs;
  ASSERT_EQ(fs.WriteFile("dir/child", "child"), 0);
  EXPECT_NE(fs.WriteFile("dir", "bad"), 0);
  EXPECT_NE(fs.WriteFile("dir/child/grandchild", "bad"), 0);
  EXPECT_EQ(Read(fs, "dir/child"), "child");
  EXPECT_EQ(fs.WriteFile("directory", "different prefix"), 0);
  EXPECT_EQ(fs.WriteFile("dir2/child", "different prefix"), 0);
  EXPECT_EQ(fs.WriteFile("DIR", "case sensitive"), 0);
  EXPECT_EQ(Read(fs, "DIR"), "case sensitive");
  // Normalization is lexical, not a traversal through existing entries.
  EXPECT_EQ(fs.WriteFile("dir/child/../sibling", "sibling"), 0);
  EXPECT_EQ(Read(fs, "dir/sibling"), "sibling");
}

TEST(VirtualFileSystem, CopiesAndMovedFromObjectsRemainUsable) {
  VirtualFileSystem fs;
  ASSERT_EQ(fs.WriteFile("a/b", "original"), 0);
  VirtualFileSystem copy = fs;
  ASSERT_EQ(copy.WriteFile("a/b", "copy"), 0);
  EXPECT_EQ(Read(fs, "a/b"), "original");
  auto &alias = fs;
  fs = alias;
  EXPECT_EQ(Read(fs, "a/b"), "original");
  VirtualFileSystem moved = std::move(copy);
  EXPECT_EQ(Read(moved, "a/b"), "copy");
  EXPECT_EQ(copy.WriteFile("new", "reused"), 0);
  fs = std::move(moved);
  EXPECT_EQ(Read(fs, "a/b"), "copy");
  EXPECT_EQ(moved.WriteFile("new", "reused"), 0);
  fs = std::move(alias);
  EXPECT_EQ(Read(fs, "a/b"), "copy");
}

TEST(VirtualFileSystem, EmptyAndBinaryFilesRoundTripThroughDisk) {
  VirtualFileSystem fs;
  const std::vector<uint8_t> binary{0, 1, 255, 0};
  ASSERT_EQ(fs.WriteFile(R"(nested\binary)", binary), 0);
  ASSERT_EQ(fs.WriteFile("empty", nullptr, 0), 0);
  EXPECT_NE(fs.WriteFile("invalid", nullptr, 1), 0);
  ASSERT_EQ(fs.WriteFile(u8"目录/文件", u8"文本"), 0);
  const auto path = std::filesystem::temp_directory_path() /
                    ("longmarch-vfs-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup{path};

  fs.SaveToDirectory(path);
  auto loaded = VirtualFileSystem::LoadDirectory(path);
  std::vector<uint8_t> bytes;
  ASSERT_EQ(loaded.ReadFile("nested/binary", bytes), 0);
  EXPECT_EQ(bytes, binary);
  EXPECT_EQ(Read(loaded, "empty"), "");
  EXPECT_EQ(Read(loaded, u8"目录\\文件"), u8"文本");
}
}  // namespace
