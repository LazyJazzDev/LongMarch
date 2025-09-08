#pragma once
#include "sparks/core/core_util.h"

namespace XH {
class CodeLines {
 public:
  CodeLines() = default;
  CodeLines(const char *code);
  CodeLines(const std::vector<uint8_t> &code_data);
  CodeLines(const std::string &code);

  CodeLines(const CD::VirtualFileSystem &vfs, const std::string &file_path);
  void InsertAfter(const CodeLines &other, const std::string &after_line);
  void InsertIndent(int num_spaces);
  operator std::string() const;
  operator bool() const;

  friend std::ostream &operator<<(std::ostream &os, const CodeLines &lines);

 private:
  std::vector<std::string> lines_;
};
}  // namespace XH
