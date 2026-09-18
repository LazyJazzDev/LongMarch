#include "sparkium/pipelines/portable/kernel_runtime.h"

#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace sparkium::portable {
namespace {

// ---------------------------------------------------------------------------
// SHA-256 (public-domain style compact implementation for cache keys).
// ---------------------------------------------------------------------------
class Sha256 {
 public:
  Sha256() {
    state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  }
  void Update(const uint8_t *data, size_t size) {
    total_ += size;
    while (size > 0) {
      const size_t take = std::min(size, 64 - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, data, take);
      buffer_size_ += take;
      data += take;
      size -= take;
      if (buffer_size_ == 64) {
        Compress(buffer_.data());
        buffer_size_ = 0;
      }
    }
  }
  std::array<uint8_t, 32> Final() {
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    Update(&pad, 1);
    pad = 0;
    while (buffer_size_ != 56)
      Update(&pad, 1);
    uint8_t length[8];
    for (int i = 0; i < 8; ++i)
      length[7 - i] = uint8_t(bits >> (i * 8));
    Update(length, 8);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 4; ++j)
        out[i * 4 + j] = uint8_t(state_[i] >> (24 - j * 8));
    return out;
  }

 private:
  void Compress(const uint8_t *block) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(block[i * 4]) << 24 | uint32_t(block[i * 4 + 1]) << 16 | uint32_t(block[i * 4 + 2]) << 8 |
             uint32_t(block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3], e = state_[4], f = state_[5],
             g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }
  static uint32_t Rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }
  std::array<uint32_t, 8> state_{};
  std::array<uint8_t, 64> buffer_{};
  size_t buffer_size_{0};
  uint64_t total_{0};
};

std::string RunCommand(const std::string &command, std::string *output) {
  std::array<char, 4096> buffer{};
  std::ostringstream collected;
  FILE *pipe = popen((command + " 2>&1").c_str(), "r");
  if (!pipe)
    throw std::runtime_error("portable kernel: failed to run: " + command);
  while (fgets(buffer.data(), buffer.size(), pipe))
    collected << buffer.data();
  const int status = pclose(pipe);
  if (output)
    *output = collected.str();
  if (status != 0)
    return collected.str() + "\n(exit status " + std::to_string(status) + ")";
  return {};
}

std::string ShellQuote(const std::string &path) {
  std::string out = "'";
  for (char c : path) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

class DlKernelLibrary : public KernelLibrary {
 public:
  DlKernelLibrary(const std::filesystem::path &library, const std::string &entry) {
    handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle_)
      throw std::runtime_error("portable kernel: dlopen failed for " + library.string() + ": " + dlerror());
    entry_ = reinterpret_cast<RenderPixelFn>(dlsym(handle_, entry.c_str()));
    if (!entry_) {
      std::string error = dlerror();
      dlclose(handle_);
      throw std::runtime_error("portable kernel: missing entry point " + entry + ": " + error);
    }
    context_slot_ = dlsym(handle_, "_ZN17sparkium_portable9g_ctx_ptrE");
    if (!context_slot_)
      context_slot_ = dlsym(handle_, "g_ctx_ptr");
  }
  ~DlKernelLibrary() override {
    if (handle_)
      dlclose(handle_);
  }
  RenderPixelFn Entry() const override {
    return entry_;
  }
  void *ContextSlot() const override {
    return context_slot_;
  }

 private:
  void *handle_{nullptr};
  RenderPixelFn entry_{nullptr};
  void *context_slot_{nullptr};
};

std::map<std::string, std::weak_ptr<KernelLibrary>> g_cache;
std::mutex g_cache_mutex;

}  // namespace

std::string HashString(const std::string &text) {
  Sha256 sha;
  sha.Update(reinterpret_cast<const uint8_t *>(text.data()), text.size());
  const auto digest = sha.Final();
  std::ostringstream out;
  for (uint8_t b : digest) {
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02x", b);
    out << buf;
  }
  return out.str();
}

std::shared_ptr<KernelLibrary> CompileKernelHost(const std::string &source,
                                                 const std::string &cache_directory,
                                                 const std::string &label) {
  const char *compiler_env = std::getenv("SPARKIUM_PORTABLE_CXX");
  // The cache key must capture everything that affects the compiled module:
  // the source, the compiler identity, and the flags (which encode the -O3
  // codegen-bug workaround). Bump the abi tag when the flags change.
  const std::string key =
      HashString(source + "\n;abi=host-v2;cxx=" + (compiler_env && *compiler_env ? compiler_env : "c++"));
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  if (auto cached = g_cache[key].lock())
    return cached;

  std::filesystem::path dir = cache_directory.empty() ? std::filesystem::temp_directory_path() / "sparkium_portable"
                                                      : std::filesystem::path(cache_directory);
  std::filesystem::create_directories(dir);
  const std::filesystem::path cpp = dir / ("kernel_" + key + ".cpp");
  const std::filesystem::path so = dir / ("kernel_" + key + ".so");

  if (!std::filesystem::exists(so)) {
    {
      std::ofstream stream(cpp);
      if (!stream)
        throw std::runtime_error("portable kernel: cannot write " + cpp.string());
      stream << source;
    }
    const char *compiler = std::getenv("SPARKIUM_PORTABLE_CXX");
    std::string cxx = compiler && *compiler ? compiler : "c++";
    // Locate hlsl_compat.h relative to this source tree (LONGMARCH_PORTABLE_INCLUDE_DIR
    // is baked in at configure time; the env var wins when set).
    const char *include_env = std::getenv("SPARKIUM_PORTABLE_INCLUDE");
    const std::string include_dir =
        include_env && *include_env ? include_env : std::string(LONGMARCH_PORTABLE_INCLUDE_DIR);
    std::ostringstream command;
    // The generated module needs sparkium_portable::g_ctx_ptr from the host
    // executable; link against it via -Wl,--unresolved-symbols=ignore-all is
    // fragile, so provide the definition inline through a force-include shim.
    const std::filesystem::path shim = dir / ("kernel_ctx_" + key + ".cpp");
    {
      std::ofstream stream(shim);
      stream << "namespace sparkium_portable {\n"
                "struct KernelContext;\n"
                "const KernelContext *g_ctx_ptr = nullptr;\n"
                "}\n";
    }
    // NOTE: -O3 is required, not merely a performance choice. GCC 12/13 on
    // x86-64 miscompile the vector/matrix sret copies in the transpiled
    // shading code at -O0/-O1/-O2 (the returned struct bytes are scattered
    // to the wrong stack slots, corrupting the caller frame); at -O3 the
    // matrices are fully scalarized and the generated code is correct. This
    // was verified against standalone reproducers. See hlsl_compat.h.
    command << cxx << " -std=c++17 -O3 -fno-fast-math -fPIC -shared -I " << ShellQuote(include_dir) << " -x c++ "
            << ShellQuote(cpp.string()) << " -x c++ " << ShellQuote(shim.string()) << " -o "
            << ShellQuote(so.string());
    std::string output;
    const auto error = RunCommand(command.str(), &output);
    if (!error.empty())
      throw std::runtime_error("portable kernel compilation failed (" + label + "):\n" + error);
  }
  auto library = std::make_shared<DlKernelLibrary>(so, "PortableRenderPixel");
  g_cache[key] = library;
  return library;
}

}  // namespace sparkium::portable
