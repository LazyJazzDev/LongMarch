#include "native_shader_compat.h"

#include <regex>
#include <stdexcept>

namespace grassland::graphics::backend {
namespace {
void Replace(std::string &s, const std::string &a, const std::string &b) {
  size_t p = 0;
  while ((p = s.find(a, p)) != std::string::npos) {
    s.replace(p, a.size(), b);
    p += b.size();
  }
}
}  // namespace

// DXC's single buffer-type templates predate Slang generics. Both native targets
// represent read-only and writable byte buffers by the same pointer/size ABI.
// Erase only these known templates, not their function bodies or algorithms.
// The original sources still compile unchanged through DXC on graphics devices.
std::string LowerLegacySource(std::string s, const std::string &filename, bool optix) {
  // Native descriptor arrays are ordinary pointer-backed arrays, so this
  // graphics descriptor-indexing annotation has no native semantic effect.
  Replace(s, "NonUniformResourceIndex(", "(");
  if (filename == "buffer_helper.hlsli") {
    auto a = s.find("template <class BufferType>\nclass BufferReference");
    auto b = s.find("template <>", a);
    auto f = s.find("template <class BufferType>\nBufferReference", a);
    if (a == std::string::npos || b == std::string::npos || f == std::string::npos)
      throw std::runtime_error("buffer helper template layout changed; update native lowering");
    auto helpers = s.substr(0, a);
    helpers = std::regex_replace(helpers, std::regex(R"(template\s*<class BufferType>\s*)"), "");
    Replace(helpers, "BufferType", "BufferReference");
    auto factory = s.substr(f, b - f);
    s.erase(a, b - a);
    auto end = s.find("template <class BufferType>\nclass StreamedBufferReference");
    s.insert(end, helpers + factory);
    s = std::regex_replace(s, std::regex(R"((  (?:float|uint|int)[234]? Load\w+\(\)))"), "  [mutating] $1");
  }
  s = std::regex_replace(s, std::regex(R"(template\s*<\s*(?:class\s+(?:BufferType|GeometrySamplerType|B))?\s*>\s*)"),
                         "");
  if (s.find("template <") != std::string::npos)
    throw std::runtime_error("unsupported native shader template in " + filename);
  Replace(s, ".template ", ".");
  s = std::regex_replace(s, std::regex(R"(\bclass\b)"), "struct");
  s = std::regex_replace(s, std::regex(R"(\bBufferType\b)"), "RWByteAddressBuffer");
  if (filename == "layout.hlsli")
    s = std::regex_replace(s, std::regex(R"(\bB\b)"), "RWByteAddressBuffer");
  Replace(s, "} random_device;", "};");
  Replace(s, "GeometrySamplerType", "GeometrySampler");
  s = std::regex_replace(s, std::regex(R"(<(?:RWByteAddressBuffer|ByteAddressBuffer)>)"), "");
  s = std::regex_replace(s, std::regex(R"(\bByteAddressBuffer\b)"), "RWByteAddressBuffer");
  Replace(s, "void SetTransform(", "[mutating] void SetTransform(");
  if (filename == "principled_bsdf.hlsli") {
    Replace(s, "void CalculateClosureWeight(", "[mutating] void CalculateClosureWeight(");
    Replace(s, "float PrincipledThinReflectionProbability(", "[mutating] float PrincipledThinReflectionProbability(");
    Replace(s, "void SamplePrincipledThinReflection(", "[mutating] void SamplePrincipledThinReflection(");
    // These methods call CalculateClosureWeight and update the local BSDF state.
    s = std::regex_replace(
        s, std::regex(R"((^|\n)(Spectrum|float3|void) (EvalPrincipledBSDF\w*|SamplePrincipledBSDF\w*)\()"),
        "$1[mutating] $2 $3(");
  }
  // Keep register-space identity as a reflected attribute on native targets.
  // Unlike hardcoded name maps, this follows conditional shader bindings.
  s = std::regex_replace(
      s,
      std::regex(
          R"(((?:cbuffer|RaytracingAccelerationStructure|(?:RW)?ByteAddressBuffer|ConstantBuffer<\w+>|(?:RW)?Texture2D<\w+>|SamplerState)\s+\w+\s*(?:\[\s*\])?)\s*:\s*register\([but s][0-9]+,\s*space([0-9]+)\))"),
      "[NativeBinding($2)] $1");
  s = std::regex_replace(s, std::regex(R"(\[\[vk::image_format\("[^"]+"\)\]\])"), "");
  for (const char *type : {"float4", "float3", "float2", "float", "int", "uint"}) {
    Replace(s, std::string("RWTexture2D<") + type + ">", std::string("NativeTexture_") + type);
    Replace(s, std::string("Texture2D<") + type + ">", std::string("NativeTexture_") + type);
  }

  Replace(s, "SamplerState", "NativeSamplerState");
  if (!optix)
    s = std::regex_replace(s, std::regex(R"(\bRayDesc\b)"), "NativeRayDesc");
  // Use an explicit descriptor span. CUDA unsized resource arrays otherwise
  // receive a reflection layout that differs from the emitted C++ structure.
  s = std::regex_replace(s,
                         std::regex(R"((RWByteAddressBuffer|NativeTexture_\w+|NativeSamplerState)\s+(\w+)\s*\[\s*\])"),
                         "NativeArray<$1> $2");
  return s;
}

std::string RenameLegacyEntry(std::string source, const std::string &entry, const std::string &replacement) {
  return std::regex_replace(source, std::regex("\\b" + entry + "\\b"), replacement);
}

std::set<std::string> LegacyConstantBlocks(const std::string &source) {
  std::set<std::string> result;
  const std::regex block(R"(cbuffer\s+(\w+)\s*\{)");
  for (auto it = std::sregex_iterator(source.begin(), source.end(), block); it != std::sregex_iterator(); ++it)
    result.insert((*it)[1]);
  return result;
}

std::string LowerLegacyHostSource(std::string text, const std::vector<std::pair<std::string, std::string>> &names) {
  text = std::regex_replace(text, std::regex(R"(\[numthreads\([^\]]*\)\])"), "");
  const std::regex block(R"(\[NativeBinding\([0-9]+\)\]\s*cbuffer\s+\w+\s*\{([^}]*)\}\s*;?)");
  std::smatch match;
  while (std::regex_search(text, match, block)) {
    std::string fields = match[1];
    fields = std::regex_replace(fields, std::regex(R"((^|;)\s*(?=\S))"), "$1 __global export __extern_cpp ");
    text.replace(match.position(), match.length(), fields);
  }
  text = std::regex_replace(text, std::regex(R"(\[NativeBinding\([0-9]+\)\])"), "__global export __extern_cpp");
  for (const auto &name : names)
    text = RenameLegacyEntry(std::move(text), name.first, name.second);
  return text;
}
}  // namespace grassland::graphics::backend
