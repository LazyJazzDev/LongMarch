#include "sparks/core/code_lines.h"

namespace sparks {

CodeLines::CodeLines(const char *code) {
  // separate code into lines
  std::string line;
  while (*code) {
    if (*code == '\n' || *code == '\r') {
      if (!line.empty()) {
        lines_.push_back(line);
        line.clear();
      }
    } else {
      line += *code;
    }
    ++code;
  }
  if (!line.empty()) {
    lines_.push_back(line);
  }
}

CodeLines::CodeLines(const std::vector<uint8_t> &code_data)
    : CodeLines(std::string(code_data.begin(), code_data.end())) {
}

CodeLines::CodeLines(const std::string &code) : CodeLines(code.c_str()) {
}

CodeLines::CodeLines(const VirtualFileSystem &vfs, const std::string &file_path) {
  std::vector<uint8_t> code_data;
  vfs.ReadFile(file_path, code_data);
  *this = CodeLines(code_data);
}

void CodeLines::InsertAfter(const CodeLines &other, const std::string &after_line) {
  // find the line after which to insert
  auto it = std::find(lines_.begin(), lines_.end(), after_line);
  if (it != lines_.end()) {
    // insert the other code lines after the found line
    lines_.insert(it + 1, other.lines_.begin(), other.lines_.end());
  } else {
    // if not found, append at the end
    lines_.insert(lines_.end(), other.lines_.begin(), other.lines_.end());
  }
}

void CodeLines::InsertIndent(int num_spaces) {
  for (auto &line : lines_) {
    // insert spaces at the beginning of each line
    line.insert(0, num_spaces, ' ');
  }
}

void CodeLines::InsertFront(const CodeLines &other) {
  // insert the other code lines at the front
  lines_.insert(lines_.begin(), other.lines_.begin(), other.lines_.end());
}

void CodeLines::InsertBack(const CodeLines &other) {
  // insert the other code lines at the back
  lines_.insert(lines_.end(), other.lines_.begin(), other.lines_.end());
}

CodeLines::operator std::string() const {
  // convert back to string
  std::string result;
  for (const auto &line : lines_) {
    result += line + "\n";
  }
  return result;
}

CodeLines::operator bool() const {
  return !lines_.empty();
}

std::ostream &operator<<(std::ostream &os, const CodeLines &lines) {
  // output the code with line numbers
  for (size_t i = 0; i < lines.lines_.size(); ++i) {
    os << std::setw(4) << i + 1 << ": " << lines.lines_[i] << "\n";
  }
  return os;
}

}  // namespace sparks
