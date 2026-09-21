#include "sparkium/backend/cpu/slang_compiler.h"

#include <fstream>

namespace sparkium::backend::cpu {
std::string Read(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

void Write(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p, std::ios::binary);
  f << s;
  if (!f)
    throw std::runtime_error("cannot write compute shader temporary file");
}

void SlangCheck(SlangResult result, const std::string &message) {
  if (SLANG_FAILED(result))
    throw std::runtime_error(message);
}

struct SlangSessionOwner {
  SlangSession *session = spCreateSession();

  SlangSessionOwner() {
    if (!session)
      throw std::runtime_error("failed to create Slang session");
  }

  ~SlangSessionOwner() {
    spDestroySession(session);
  }
};

SlangSession *Session() {
  static SlangSessionOwner owner;
  return owner.session;
}

}  // namespace sparkium::backend::cpu
