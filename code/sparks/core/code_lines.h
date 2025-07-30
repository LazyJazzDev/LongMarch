#pragma once
#include "sparks/core/core_util.h"

namespace sparks {
class CodeLines {
 public:
  CodeLines() = default;
  CodeLines(const char *code);
  CodeLines(const std::vector<uint8_t> &code_data);
  CodeLines(const std::string &code);
  void InsertAfter(const CodeLines &other, const std::string &after_line);
  operator std::string() const;
  void InsertIndent(int num_spaces);

 private:
  std::vector<std::string> lines_;
};
}  // namespace sparks
