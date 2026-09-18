#include "sparkium/scene_io/json_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/istreamwrapper.h"
#include "sparkium/core/core.h"

namespace sparkium {
namespace {
using Value = rapidjson::Value;

const Value &RequireObject(const Value &value) {
  if (!value.IsObject())
    throw std::runtime_error("expected a JSON object");
  return value;
}
const Value &RequireArray(const Value &value) {
  if (!value.IsArray())
    throw std::runtime_error("expected a JSON array");
  return value;
}
std::string ReadString(const Value &value) {
  if (!value.IsString())
    throw std::runtime_error("expected a string");
  return value.GetString();
}
int ReadInt(const Value &value) {
  if (!value.IsInt())
    throw std::runtime_error("expected a 32-bit integer");
  return value.GetInt();
}
float ReadNumber(const Value &value) {
  if (!value.IsNumber())
    throw std::runtime_error("expected a number");
  float result = value.GetFloat();
  if (!std::isfinite(result))
    throw std::runtime_error("expected a finite float");
  return result;
}
bool ReadBool(const Value &value) {
  if (!value.IsBool())
    throw std::runtime_error("expected a boolean");
  return value.GetBool();
}

const Value &Member(const Value &value, const char *name) {
  if (!value.IsObject() || !value.HasMember(name))
    throw std::runtime_error(std::string("missing field '") + name + "'");
  return value[name];
}

glm::vec3 Vec3(const Value &value, const glm::vec3 &fallback = {}) {
  if (value.IsNull())
    return fallback;
  if (!value.IsArray() || value.Size() != 3)
    throw std::runtime_error("expected an array of three numbers");
  return {ReadNumber(value[0]), ReadNumber(value[1]), ReadNumber(value[2])};
}

glm::vec3 Vec3Member(const Value &value, const char *name, const glm::vec3 &fallback = {}) {
  return RequireObject(value).HasMember(name) ? Vec3(value[name], fallback) : fallback;
}

float FloatMember(const Value &value, const char *name, float fallback) {
  return RequireObject(value).HasMember(name) ? ReadNumber(value[name]) : fallback;
}

bool BoolMember(const Value &value, const char *name, bool fallback) {
  return RequireObject(value).HasMember(name) ? ReadBool(value[name]) : fallback;
}

std::filesystem::path Resolve(const std::filesystem::path &document, const std::string &asset) {
  std::filesystem::path path(asset);
  if (path.is_relative())
    path = document.parent_path() / path;
  return std::filesystem::weakly_canonical(path);
}

glm::mat4 Transform(const Value &value) {
  glm::mat4 result{1.0f};
  RequireObject(value);
  if (value.HasMember("matrix")) {
    const auto &matrix = value["matrix"];
    if (!matrix.IsArray() || matrix.Size() != 16)
      throw std::runtime_error("transform.matrix must contain 16 numbers");
    for (rapidjson::SizeType i = 0; i < 16; ++i)
      result[i / 4][i % 4] = ReadNumber(matrix[i]);
    return result;
  }
  result = glm::translate(result, Vec3Member(value, "translation"));
  auto rotation = glm::radians(Vec3Member(value, "rotation_degrees"));
  result = glm::rotate(result, rotation.x, {1.0f, 0.0f, 0.0f});
  result = glm::rotate(result, rotation.y, {0.0f, 1.0f, 0.0f});
  result = glm::rotate(result, rotation.z, {0.0f, 0.0f, 1.0f});
  result = glm::scale(result, Vec3Member(value, "scale", {1.0f, 1.0f, 1.0f}));
  return result;
}

Mesh<> InlineMesh(const Value &value) {
  const auto &positions_json = RequireArray(Member(value, "positions"));
  const auto &indices_json = RequireArray(Member(value, "indices"));
  std::vector<Vector3<float>> positions;
  std::vector<Vector2<float>> tex_coords;
  std::vector<uint32_t> indices;
  for (const auto &item : positions_json.GetArray()) {
    auto p = Vec3(item);
    positions.push_back({p.x, p.y, p.z});
  }
  for (const auto &item : indices_json.GetArray()) {
    if (!item.IsUint() || item.GetUint() >= positions.size())
      throw std::runtime_error("mesh index must reference an existing vertex");
    indices.push_back(item.GetUint());
  }
  if (positions.empty() || indices.empty() || indices.size() % 3 != 0)
    throw std::runtime_error("inline mesh must contain vertices and complete triangles");
  if (value.HasMember("tex_coords")) {
    for (const auto &item : RequireArray(value["tex_coords"]).GetArray()) {
      if (!item.IsArray() || item.Size() != 2)
        throw std::runtime_error("texture coordinate must have two numbers");
      tex_coords.push_back({ReadNumber(item[0]), ReadNumber(item[1])});
    }
  }
  if (value.HasMember("tex_coords") && tex_coords.size() != positions.size())
    throw std::runtime_error("mesh texture coordinates must match vertex count");
  return Mesh<>(positions.size(), indices.size(), indices.data(), positions.data(), nullptr,
                tex_coords.empty() ? nullptr : tex_coords.data());
}

Mesh<> BinaryMesh(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open binary mesh: " + path.string());
  const auto size = stream.tellg();
  stream.seekg(0);
  std::array<char, 8> magic{};
  uint32_t num_vertices = 0, num_indices = 0, flags = 0;
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&num_vertices), sizeof(num_vertices));
  stream.read(reinterpret_cast<char *>(&num_indices), sizeof(num_indices));
  stream.read(reinterpret_cast<char *>(&flags), sizeof(flags));
  if (!stream || std::memcmp(magic.data(), "SPKMESH1", 8) != 0)
    throw std::runtime_error("invalid Sparkium binary mesh: " + path.string());
  if (num_indices % 3 != 0 || (flags & ~7u))
    throw std::runtime_error("invalid Sparkium binary mesh header: " + path.string());
  const uint64_t expected =
      20ull + uint64_t(num_indices) * sizeof(uint32_t) +
      uint64_t(num_vertices) * (sizeof(float) * 3 + ((flags & 1u) ? sizeof(float) * 3 : 0) +
                                ((flags & 2u) ? sizeof(float) * 2 : 0) + ((flags & 4u) ? sizeof(float) * 3 : 0));
  if (uint64_t(size) != expected)
    throw std::runtime_error("truncated or oversized Sparkium binary mesh: " + path.string());
  std::vector<uint32_t> indices(num_indices);
  std::vector<Vector3<float>> positions(num_vertices), normals;
  std::vector<Vector2<float>> tex_coords;
  std::vector<Vector3<float>> colors;
  stream.read(reinterpret_cast<char *>(indices.data()), indices.size() * sizeof(uint32_t));
  stream.read(reinterpret_cast<char *>(positions.data()), positions.size() * sizeof(Vector3<float>));
  if (flags & 1u) {
    normals.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(normals.data()), normals.size() * sizeof(Vector3<float>));
  }
  if (flags & 2u) {
    tex_coords.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(tex_coords.data()), tex_coords.size() * sizeof(Vector2<float>));
  }
  if (flags & 4u) {
    colors.resize(num_vertices);
    stream.read(reinterpret_cast<char *>(colors.data()), colors.size() * sizeof(Vector3<float>));
  }
  if (!stream)
    throw std::runtime_error("cannot read binary mesh: " + path.string());
  return Mesh<>(positions.size(), indices.size(), indices.data(), positions.data(),
                normals.empty() ? nullptr : normals.data(), tex_coords.empty() ? nullptr : tex_coords.data(), nullptr,
                colors.empty() ? nullptr : colors.data());
}

struct BinaryHairData {
  std::vector<Vector3<float>> points;
  std::vector<float> radii;
  std::vector<uint32_t> offsets;
};

BinaryHairData BinaryHair(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open binary hair: " + path.string());
  const auto size = stream.tellg();
  stream.seekg(0);
  std::array<char, 8> magic{};
  uint32_t strand_count = 0, point_count = 0;
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char *>(&strand_count), sizeof(strand_count));
  stream.read(reinterpret_cast<char *>(&point_count), sizeof(point_count));
  const uint64_t expected = 16ull + uint64_t(strand_count + 1) * sizeof(uint32_t) +
                            uint64_t(point_count) * sizeof(Vector3<float>) + uint64_t(point_count) * sizeof(float);
  if (!stream || std::memcmp(magic.data(), "SPKHAIR1", 8) != 0 || uint64_t(size) != expected)
    throw std::runtime_error("invalid Sparkium binary hair: " + path.string());
  BinaryHairData result;
  result.offsets.resize(strand_count + 1);
  result.points.resize(point_count);
  result.radii.resize(point_count);
  stream.read(reinterpret_cast<char *>(result.offsets.data()), result.offsets.size() * sizeof(uint32_t));
  stream.read(reinterpret_cast<char *>(result.points.data()), result.points.size() * sizeof(Vector3<float>));
  stream.read(reinterpret_cast<char *>(result.radii.data()), result.radii.size() * sizeof(float));
  if (!stream || result.offsets.empty() || result.offsets.front() != 0 || result.offsets.back() != point_count)
    throw std::runtime_error("invalid Sparkium hair offsets: " + path.string());
  return result;
}

// Compiles a JSON shader-node graph into two equivalent forms in a single
// traversal: the HLSL used by the graphics pipelines, and the backend-neutral
// instruction list (`ShaderGraphProgram`) interpreted by the native CPU/CUDA
// backends. Every node emits one HLSL assignment and one instruction with the
// same operands, so the two forms cannot drift apart.
class ShaderGraphCompiler {
 public:
  ShaderGraphCompiler(const Value &graph, const std::map<std::string, int> &texture_slots)
      : graph_(graph), nodes_(Member(graph, "nodes")), texture_slots_(texture_slots) {
    if (!nodes_.IsObject())
      throw std::runtime_error("shader graph nodes must be an object");
  }

  CodeLines Compile() {
    lines_ << R"(
float GraphHash(float3 p) {
  p = frac(p * 0.3183099f + 0.1f); p *= 17.0f;
  return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float GraphNoise(float3 p) {
  float3 i = floor(p), f = frac(p); f = f * f * (3.0f - 2.0f * f);
  float n000=GraphHash(i), n100=GraphHash(i+float3(1,0,0));
  float n010=GraphHash(i+float3(0,1,0)), n110=GraphHash(i+float3(1,1,0));
  float n001=GraphHash(i+float3(0,0,1)), n101=GraphHash(i+float3(1,0,1));
  float n011=GraphHash(i+float3(0,1,1)), n111=GraphHash(i+float3(1,1,1));
  return lerp(lerp(lerp(n000,n100,f.x),lerp(n010,n110,f.x),f.y),
              lerp(lerp(n001,n101,f.x),lerp(n011,n111,f.x),f.y),f.z);
}
float3 GraphSrgbToLinear(float3 color) {
  float3 lower=color/12.92f;
  float3 upper=pow((color+0.055f)/1.055f,2.4f);
  return lerp(lower,upper,step(0.04045f,color));
}
float GraphVoronoi(float3 p) {
  float3 cell=floor(p), local=frac(p); float distance=1e9f;
  for(int z=-1;z<=1;++z) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
    float3 o=float3(x,y,z), sample_point=o+GraphHash(cell+o+float3(0,0,0))*float3(0.73f,0.91f,0.57f);
    distance=min(distance,length(sample_point-local));
  }
  return distance;
}
float3 GraphRotateXYZ(float3 p, float3 r) {
  float3 s=sin(r), c=cos(r);
  p=float3(p.x, c.x*p.y-s.x*p.z, s.x*p.y+c.x*p.z);
  p=float3(c.y*p.x+s.y*p.z, p.y, -s.y*p.x+c.y*p.z);
  return float3(c.z*p.x-s.z*p.y, s.z*p.x+c.z*p.y, p.z);
}
float3 GraphRgbToHsv(float3 c) {
  float4 K=float4(0.0f,-1.0f/3.0f,2.0f/3.0f,-1.0f);
  float4 p=lerp(float4(c.bg,K.wz),float4(c.gb,K.xy),step(c.b,c.g));
  float4 q=lerp(float4(p.xyw,c.r),float4(c.r,p.yzx),step(p.x,c.r));
  float d=q.x-min(q.w,q.y), e=1.0e-10f;
  return float3(abs(q.z+(q.w-q.y)/(6.0f*d+e)),d/(q.x+e),q.x);
}
float3 GraphHsvToRgb(float3 c) {
  float3 p=abs(frac(c.xxx+float3(0.0f,2.0f/3.0f,1.0f/3.0f))*6.0f-3.0f);
  return c.z*lerp(float3(1,1,1),saturate(p-1.0f),c.y);
}
GraphSurface EvaluateShaderGraph(HitRecord hit_record, float3 view_direction, int bounce, int ray_type, bool is_shadow_ray,
                                 ByteAddressBuffer material_data) {
)";
    const auto &surface = RequireObject(Member(graph_, "surface"));
    const std::pair<const char *, std::string> outputs[] = {
        {"base_color", "surface.base_color"},
        {"metallic", "surface.metallic"},
        {"specular", "surface.specular"},
        {"roughness", "surface.roughness"},
        {"anisotropic", "surface.anisotropic"},
        {"anisotropic_rotation", "surface.anisotropic_rotation"},
        {"sheen", "surface.sheen"},
        {"clearcoat", "surface.clearcoat"},
        {"clearcoat_roughness", "surface.clearcoat_roughness"},
        {"ior", "surface.ior"},
        {"transmission", "surface.transmission"},
        {"transmission_roughness", "surface.transmission_roughness"},
        {"emission", "surface.emission"},
        {"normal", "surface.normal"},
        {"opacity", "surface.opacity"},
        {"shadow_opacity", "surface.shadow_opacity"},
        {"thin_walled", "surface.thin_walled"},
        {"subsurface", "surface.subsurface"},
        {"subsurface_scale", "surface.subsurface_scale"},
        {"subsurface_radius", "surface.subsurface_radius"},
        {"subsurface_method", "surface.subsurface_method"}};
    static_assert(sizeof(outputs) / sizeof(outputs[0]) == kShaderGraphSurfaceOutputCount,
                  "the HLSL and native surface output tables must stay in sync");
    lines_ << "  GraphSurface surface;\n"
           << "  surface.base_color=float3(0.8f,0.8f,0.8f); "
              "surface.metallic=0.0f; surface.specular=0.5f;\n"
           << "  surface.roughness=0.5f; surface.anisotropic=0.0f; "
              "surface.anisotropic_rotation=0.0f;\n"
           << "  surface.sheen=0.0f; surface.clearcoat=0.0f; "
              "surface.clearcoat_roughness=0.03f;\n"
           << "  surface.ior=1.45f; surface.transmission=0.0f; "
              "surface.transmission_roughness=0.0f;\n"
           << "  surface.emission=float3(0,0,0); "
              "surface.normal=hit_record.normal; surface.opacity=1.0f; "
              "surface.shadow_opacity=-1.0f;\n"
           << "  surface.thin_walled=0.0f; surface.subsurface=0.0f; "
              "surface.subsurface_scale=0.0f;\n"
           << "  surface.subsurface_radius=float3(1.0f,0.2f,0.1f); "
              "surface.subsurface_method=0.0f;\n";
    std::vector<std::pair<std::string, std::string>> assignments;
    for (int index = 0; index < kShaderGraphSurfaceOutputCount; ++index) {
      const auto &[name, target] = outputs[index];
      if (!surface.HasMember(name))
        continue;
      auto expression = Expression(surface[name]);
      program_.outputs[index] = expression.slot;
      assignments.emplace_back(target,
                               expression.code + ((std::string(name) == "base_color" || std::string(name) == "emission" ||
                                              std::string(name) == "normal" || std::string(name) == "subsurface_radius")
                                                 ? ".xyz"
                                                 : ".x"));
    }
    for (const auto &[target, expression] : assignments)
      lines_ << "  " << target << "=" << expression << ";\n";
    lines_ << "  return surface;\n}\n";
    AllocateRegisters();
    return CodeLines(lines_.str());
  }

  // Valid once `Compile()` has run.
  const ShaderGraphProgram &Program() const {
    return program_;
  }

 private:
  // An evaluated graph input: the HLSL expression and the operand-encoded
  // source of the same value in the instruction list.
  struct Expr {
    std::string code;
    int32_t slot{kShaderGraphOperandNone};
  };

  std::string Number(double value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    auto text = out.str();
    if (text.find_first_of(".eE") == std::string::npos)
      text += ".0";
    return text + "f";
  }

  int32_t Constant(const glm::vec4 &value) {
    for (size_t i = 0; i < program_.constants.size(); ++i)
      if (program_.constants[i] == value)
        return -static_cast<int32_t>(i) - 2;
    program_.constants.push_back(value);
    return -static_cast<int32_t>(program_.constants.size() - 1) - 2;
  }

  Expr Const(const char *code, const glm::vec4 &value) {
    return {code, Constant(value)};
  }

  int32_t Emit(ShaderGraphOp op,
               uint32_t sub,
               std::initializer_list<int32_t> operands,
               uint32_t data_offset = 0,
               uint32_t data_count = 0) {
    ShaderGraphInstruction instruction;
    instruction.op = op;
    instruction.sub = sub;
    int index = 0;
    for (int32_t operand : operands) {
      if (index >= kShaderGraphMaxOperands)
        throw std::runtime_error("shader graph node uses too many operands");
      instruction.operands[index++] = operand;
    }
    instruction.data_offset = data_offset;
    instruction.data_count = data_count;
    // Values are numbered by definition order while the program is built;
    // `AllocateRegisters()` maps them onto the device register file.
    instruction.dst = static_cast<int32_t>(program_.instructions.size());
    program_.instructions.push_back(instruction);
    return instruction.dst;
  }

  // Reuses a device register as soon as the value it holds is dead. Graphs in
  // the shipped scenes reach fifty nodes but only a handful of live values.
  void AllocateRegisters() {
    const size_t count = program_.instructions.size();
    std::vector<int> last_use(count);
    for (size_t i = 0; i < count; ++i)
      last_use[i] = static_cast<int>(i);
    for (size_t i = 0; i < count; ++i)
      for (int32_t operand : program_.instructions[i].operands)
        if (operand >= 0)
          last_use[operand] = static_cast<int>(i);
    for (int32_t slot : program_.outputs)
      if (slot >= 0)
        last_use[slot] = static_cast<int>(count);
    std::vector<int> physical(count, -1), free_list;
    int next_register = 0;
    for (size_t i = 0; i < count; ++i) {
      int destination;
      if (!free_list.empty()) {
        destination = free_list.back();
        free_list.pop_back();
      } else {
        destination = next_register++;
        if (next_register > kShaderGraphMaxRegisters)
          throw std::runtime_error("shader graph needs more registers than the native backends provide");
      }
      physical[i] = destination;
      std::set<int32_t> released;
      for (int32_t operand : program_.instructions[i].operands)
        if (operand >= 0 && last_use[operand] == static_cast<int>(i) && released.insert(operand).second)
          free_list.push_back(physical[operand]);
      if (last_use[i] == static_cast<int>(i))
        free_list.push_back(destination);
    }
    for (size_t i = 0; i < count; ++i) {
      auto &instruction = program_.instructions[i];
      instruction.dst = physical[i];
      for (int32_t &operand : instruction.operands)
        if (operand >= 0)
          operand = physical[operand];
    }
    for (int32_t &slot : program_.outputs)
      if (slot >= 0)
        slot = physical[slot];
    program_.register_count = next_register;
  }

  glm::vec4 LiteralValue(const Value &value) {
    if (value.IsNumber()) {
      float scalar = ReadNumber(value);
      return {scalar, scalar, scalar, scalar};
    }
    std::vector<float> parts;
    for (const auto &item : value.GetArray())
      parts.push_back(ReadNumber(item));
    while (parts.size() < 4)
      parts.push_back(parts.size() == 3 ? 1.0f : parts.back());
    return {parts[0], parts[1], parts[2], parts[3]};
  }

  Expr Literal(const Value &value) {
    if (value.IsNumber()) {
      auto n = Number(ReadNumber(value));
      return {"float4(" + n + "," + n + "," + n + "," + n + ")", Constant(LiteralValue(value))};
    }
    if (!value.IsArray() || value.Empty() || value.Size() > 4)
      throw std::runtime_error("shader input must be a number, vector, or node reference");
    std::vector<std::string> parts;
    for (const auto &item : value.GetArray())
      parts.push_back(Number(ReadNumber(item)));
    while (parts.size() < 4)
      parts.push_back(parts.size() == 3 ? "1.0f" : parts.back());
    return {"float4(" + parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + ")",
            Constant(LiteralValue(value))};
  }

  Expr Input(const Value &node, const char *name, const Expr &fallback) {
    if (!node.HasMember("inputs") || !RequireObject(node["inputs"]).HasMember(name))
      return fallback;
    return Expression(node["inputs"][name]);
  }

  // Fallbacks that read the hit record have to be emitted as instructions, so
  // they are only built when the input is actually missing.
  template <typename Fallback>
  Expr InputLazy(const Value &node, const char *name, Fallback &&fallback) {
    if (!node.HasMember("inputs") || !RequireObject(node["inputs"]).HasMember(name))
      return fallback();
    return Expression(node["inputs"][name]);
  }

  Expr TexCoord() {
    return {"float4(hit_record.tex_coord,0,0)", Emit(SHADER_GRAPH_OP_TEXTURE_COORDINATE, 0, {})};
  }
  Expr ShadingNormal() {
    return {"float4(hit_record.normal,0)", Emit(SHADER_GRAPH_OP_TEXTURE_COORDINATE, 1, {})};
  }
  Expr ObjectPosition() {
    return {"float4(hit_record.object_position,0)", Emit(SHADER_GRAPH_OP_TEXTURE_COORDINATE, 2, {})};
  }

  uint32_t PushData(const std::vector<float> &values) {
    const auto offset = static_cast<uint32_t>(program_.data.size());
    program_.data.insert(program_.data.end(), values.begin(), values.end());
    return offset;
  }

  Expr Expression(const Value &value) {
    if (!value.IsObject() || !value.HasMember("node"))
      return Literal(value);
    return Node(ReadString(value["node"]), value.HasMember("output") ? ReadString(value["output"]) : "value");
  }

  Expr Node(const std::string &id, const std::string &output) {
    const auto key = id + ":" + output;
    if (auto it = cache_.find(key); it != cache_.end())
      return it->second;
    if (!nodes_.HasMember(id.c_str()))
      throw std::runtime_error("shader graph references unknown node: " + id);
    if (!active_.insert(key).second)
      throw std::runtime_error("shader graph contains a cycle at node: " + id);
    const auto &node = RequireObject(nodes_[id.c_str()]);
    const std::string type = ReadString(Member(node, "type"));
    const std::string var = "graph_v" + std::to_string(next_variable_++);
    std::string expression;
    int32_t slot = kShaderGraphOperandNone;
    const Expr zero = Const("float4(0,0,0,0)", {0.0f, 0.0f, 0.0f, 0.0f});
    const Expr one = Const("float4(1,1,1,1)", {1.0f, 1.0f, 1.0f, 1.0f});
    const Expr half = Const("float4(0.5,0.5,0.5,0.5)", {0.5f, 0.5f, 0.5f, 0.5f});
    const Expr five = Const("float4(5,5,5,5)", {5.0f, 5.0f, 5.0f, 5.0f});
    if (type == "value" || type == "rgb") {
      auto value = Input(node, "value", zero);
      expression = value.code;
      slot = Emit(SHADER_GRAPH_OP_COPY, 0, {value.slot});
    } else if (type == "texture_coordinate") {
      uint32_t sub = 2;
      if (output == "uv") {
        expression = "float4(hit_record.tex_coord,0,0)";
        sub = 0;
      } else if (output == "normal") {
        expression = "float4(hit_record.normal,0)";
        sub = 1;
      } else {
        expression = "float4(hit_record.object_position,0)";
      }
      slot = Emit(SHADER_GRAPH_OP_TEXTURE_COORDINATE, sub, {});
    } else if (type == "image_texture") {
      auto it = texture_slots_.find(id);
      if (it == texture_slots_.end())
        throw std::runtime_error("image node has no loaded texture: " + id);
      auto vector = InputLazy(node, "vector", [&] { return TexCoord(); });
      expression =
          "SampleTexture(material_data.Load(" + std::to_string(12 + it->second * 4) + "),(" + vector.code + ").xy)";
      uint32_t sub = 0;
      if (output == "alpha") {
        expression = "(" + expression + ").wwww";
        sub = 1;
      } else if (node.HasMember("color_space") && std::string(ReadString(node["color_space"])) == "srgb") {
        expression = "float4(GraphSrgbToLinear((" + expression + ").xyz),(" + expression + ").w)";
        sub = 2;
      }
      slot = Emit(SHADER_GRAPH_OP_IMAGE_TEXTURE, sub, {vector.slot}, static_cast<uint32_t>(it->second));
    } else if (type == "mapping") {
      auto vector = Input(node, "vector", zero), location = Input(node, "location", zero);
      auto rotation = Input(node, "rotation", zero), scale = Input(node, "scale", one);
      expression = "float4(GraphRotateXYZ((" + vector.code + ").xyz*(" + scale.code + ").xyz,(" + rotation.code +
                   ").xyz)+(" + location.code + ").xyz,0)";
      slot = Emit(SHADER_GRAPH_OP_MAPPING, 0, {vector.slot, location.slot, rotation.slot, scale.slot});
    } else if (type == "noise_texture") {
      auto vector = InputLazy(node, "vector", [&] { return ObjectPosition(); });
      auto scale = Input(node, "scale", five);
      expression = "GraphNoise((" + vector.code + ").xyz*(" + scale.code + ").x).xxxx";
      slot = Emit(SHADER_GRAPH_OP_NOISE_TEXTURE, 0, {vector.slot, scale.slot});
    } else if (type == "voronoi_texture") {
      auto vector = InputLazy(node, "vector", [&] { return ObjectPosition(); });
      auto scale = Input(node, "scale", five);
      expression = "GraphVoronoi((" + vector.code + ").xyz*(" + scale.code + ").x).xxxx";
      slot = Emit(SHADER_GRAPH_OP_VORONOI_TEXTURE, 0, {vector.slot, scale.slot});
    } else if (type == "gradient_texture") {
      auto vector = InputLazy(node, "vector", [&] { return ObjectPosition(); });
      expression = "(" + vector.code + ").xxxx";
      slot = Emit(SHADER_GRAPH_OP_GRADIENT_TEXTURE, 0, {vector.slot});
    } else if (type == "wave_texture") {
      auto vector = InputLazy(node, "vector", [&] { return ObjectPosition(); });
      auto scale = Input(node, "scale", five);
      auto distortion = Input(node, "distortion", zero), phase = Input(node, "phase", zero);
      std::string wave_type = node.HasMember("wave_type") ? ReadString(node["wave_type"]) : "BANDS";
      std::string direction = node.HasMember("bands_direction") ? ReadString(node["bands_direction"]) : "X";
      std::string p = "(" + vector.code + ").xyz*(" + scale.code + ").x";
      std::string coordinate;
      uint32_t sub = 0;
      if (wave_type == "RINGS") {
        coordinate = "length(" + p + ")";
        sub = 4;
      } else if (direction == "Y") {
        coordinate = "(" + p + ").y";
        sub = 1;
      } else if (direction == "Z") {
        coordinate = "(" + p + ").z";
        sub = 2;
      } else if (direction == "DIAGONAL") {
        coordinate = "((" + p + ").x+(" + p + ").y+(" + p + ").z)*0.577350269f";
        sub = 3;
      } else {
        coordinate = "(" + p + ").x";
      }
      coordinate = "(" + coordinate + "+(" + distortion.code + ").x*GraphNoise(" + p + ")+(" + phase.code + ").x)";
      expression = "(0.5f+0.5f*sin(" + coordinate + "*6.283185307f)).xxxx";
      slot = Emit(SHADER_GRAPH_OP_WAVE_TEXTURE, sub, {vector.slot, scale.slot, distortion.slot, phase.slot});
    } else if (type == "sky_texture") {
      auto vector = Input(node, "vector", Const("float4(0,0,1,0)", {0.0f, 0.0f, 1.0f, 0.0f}));
      auto sun = node.HasMember("sun_direction") ? Literal(node["sun_direction"])
                                                 : Const("float4(0,1,0,0)", {0.0f, 1.0f, 0.0f, 0.0f});
      expression =
          "float4(lerp(float3(0.08f,0.16f,0.35f),float3(0.65f,0.78f,1."
          "0f),saturate(normalize((" +
          vector.code + ").xyz).z*0.5f+0.5f))+pow(saturate(dot(normalize((" + vector.code + ").xyz),normalize((" +
          sun.code + ").xyz))),512.0f)*float3(8,6,3),1)";
      slot = Emit(SHADER_GRAPH_OP_SKY_TEXTURE, 0, {vector.slot, sun.slot});
    } else if (type == "vertex_attribute") {
      uint32_t sub = 0;
      if (output == "alpha") {
        expression = "float4(1,1,1,1)";
        sub = 2;
      } else if (output == "factor") {
        expression = "hit_record.color.xxxx";
        sub = 1;
      } else {
        expression = "float4(hit_record.color,1)";
      }
      slot = Emit(SHADER_GRAPH_OP_VERTEX_ATTRIBUTE, sub, {});
    } else if (type == "object_info") {
      uint32_t sub = 3;
      if (output == "location") {
        expression = "float4(hit_record.object_origin,0)";
        sub = 0;
      } else if (output == "object_index" || output == "material_index") {
        expression = "float(hit_record.object_index).xxxx";
        sub = 1;
      } else if (output == "random") {
        expression = "GraphHash(float3(hit_record.object_index,17,31)).xxxx";
        sub = 2;
      } else {
        expression = "float4(1,1,1,1)";
      }
      slot = Emit(SHADER_GRAPH_OP_OBJECT_INFO, sub, {});
    } else if (type == "geometry_info") {
      uint32_t sub = 4;
      if (output == "position") {
        expression = "float4(hit_record.position,1)";
        sub = 0;
      } else if (output == "normal" || output == "true_normal") {
        expression = "float4(hit_record.normal,0)";
        sub = 1;
      } else if (output == "incoming") {
        expression = "float4(-view_direction,0)";
        sub = 2;
      } else if (output == "backfacing") {
        expression = "(hit_record.front_facing?0.0f:1.0f).xxxx";
        sub = 3;
      } else {
        expression = zero.code;
      }
      slot = Emit(SHADER_GRAPH_OP_GEOMETRY_INFO, sub, {});
    } else if (type == "light_path") {
      uint32_t sub = SHADER_GRAPH_LIGHT_PATH_ZERO;
      if (output == "is_camera_ray") {
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_CAMERA?1.0f:0.0f).xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_IS_CAMERA;
      } else if (output == "is_shadow_ray") {
        expression = "(is_shadow_ray?1.0f:0.0f).xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_IS_SHADOW;
      } else if (output == "is_reflection_ray") {
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_REFLECTION?1.0f:0.0f).xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_IS_REFLECTION;
      } else if (output == "is_transmission_ray") {
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_TRANSMISSION?1.0f:0.0f).xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_IS_TRANSMISSION;
      } else if (output == "is_glossy_ray" || output == "is_diffuse_ray" || output == "is_singular_ray") {
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_REFLECTION?1.0f:0.0f).xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_IS_REFLECTION;
      } else if (output == "ray_length") {
        expression = "hit_record.t.xxxx";
        sub = SHADER_GRAPH_LIGHT_PATH_RAY_LENGTH;
      } else {
        expression = zero.code;
      }
      slot = Emit(SHADER_GRAPH_OP_LIGHT_PATH, sub, {});
    } else if (type == "invert") {
      auto color = Input(node, "color", zero);
      auto factor = Input(node, "factor", one);
      expression = "lerp(" + color.code + ",1.0f-" + color.code + ",(" + factor.code + ").x)";
      slot = Emit(SHADER_GRAPH_OP_INVERT, 0, {color.slot, factor.slot});
    } else if (type == "mix") {
      auto factor = Input(node, "factor", half);
      auto a = Input(node, "a", zero), b = Input(node, "b", one);
      std::string blend = node.HasMember("blend_type") ? ReadString(node["blend_type"]) : "MIX";
      ShaderGraphBlendType blend_type = SHADER_GRAPH_BLEND_MIX;
      if (blend == "MULTIPLY") {
        expression = "lerp(" + a.code + "," + a.code + "*" + b.code + ",saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_MULTIPLY;
      } else if (blend == "ADD") {
        expression = a.code + "+" + b.code + "*(" + factor.code + ").x";
        blend_type = SHADER_GRAPH_BLEND_ADD;
      } else if (blend == "SUBTRACT") {
        expression = a.code + "-" + b.code + "*(" + factor.code + ").x";
        blend_type = SHADER_GRAPH_BLEND_SUBTRACT;
      } else if (blend == "DARKEN") {
        expression = "lerp(" + a.code + ",min(" + a.code + "," + b.code + "),saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_DARKEN;
      } else if (blend == "LIGHTEN") {
        expression = "lerp(" + a.code + ",max(" + a.code + "," + b.code + "),saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_LIGHTEN;
      } else if (blend == "SCREEN") {
        expression = "lerp(" + a.code + ",1.0f-(1.0f-" + a.code + ")*(1.0f-" + b.code + "),saturate((" + factor.code +
                     ").x))";
        blend_type = SHADER_GRAPH_BLEND_SCREEN;
      } else if (blend == "DIFFERENCE") {
        expression = "lerp(" + a.code + ",abs(" + a.code + "-" + b.code + "),saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_DIFFERENCE;
      } else if (blend == "DIVIDE") {
        expression =
            "lerp(" + a.code + "," + a.code + "/max(abs(" + b.code + "),1e-8f),saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_DIVIDE;
      } else if (blend == "OVERLAY") {
        auto overlay = "lerp(2.0f*" + a.code + "*" + b.code + ",1.0f-2.0f*(1.0f-" + a.code + ")*(1.0f-" + b.code +
                       "),step(0.5f," + a.code + "))";
        expression = "lerp(" + a.code + "," + overlay + ",saturate((" + factor.code + ").x))";
        blend_type = SHADER_GRAPH_BLEND_OVERLAY;
      } else {
        expression = "lerp(" + a.code + "," + b.code + ",saturate((" + factor.code + ").x))";
      }
      slot = Emit(SHADER_GRAPH_OP_MIX, blend_type, {factor.slot, a.slot, b.slot});
    } else if (type == "math") {
      auto a = Input(node, "a", zero), b = Input(node, "b", zero), c = Input(node, "c", zero);
      std::string op = node.HasMember("operation") ? ReadString(node["operation"]) : "ADD";
      std::string x = "(" + a.code + ").x", y = "(" + b.code + ").x", z = "(" + c.code + ").x", scalar;
      ShaderGraphMathOp math_op = SHADER_GRAPH_MATH_ADD;
      if (op == "MULTIPLY") {
        scalar = x + "*" + y;
        math_op = SHADER_GRAPH_MATH_MULTIPLY;
      } else if (op == "SUBTRACT") {
        scalar = x + "-" + y;
        math_op = SHADER_GRAPH_MATH_SUBTRACT;
      } else if (op == "DIVIDE") {
        scalar = x + "/max(abs(" + y + "),1e-8f)";
        math_op = SHADER_GRAPH_MATH_DIVIDE;
      } else if (op == "POWER") {
        scalar = "pow(max(" + x + ",0.0f)," + y + ")";
        math_op = SHADER_GRAPH_MATH_POWER;
      } else if (op == "MINIMUM") {
        scalar = "min(" + x + "," + y + ")";
        math_op = SHADER_GRAPH_MATH_MINIMUM;
      } else if (op == "MAXIMUM") {
        scalar = "max(" + x + "," + y + ")";
        math_op = SHADER_GRAPH_MATH_MAXIMUM;
      } else if (op == "LESS_THAN") {
        scalar = "(" + x + "<" + y + "?1.0f:0.0f)";
        math_op = SHADER_GRAPH_MATH_LESS_THAN;
      } else if (op == "GREATER_THAN") {
        scalar = "(" + x + ">" + y + "?1.0f:0.0f)";
        math_op = SHADER_GRAPH_MATH_GREATER_THAN;
      } else if (op == "MULTIPLY_ADD") {
        scalar = x + "*" + y + "+" + z;
        math_op = SHADER_GRAPH_MATH_MULTIPLY_ADD;
      } else if (op == "ABSOLUTE") {
        scalar = "abs(" + x + ")";
        math_op = SHADER_GRAPH_MATH_ABSOLUTE;
      } else if (op == "SINE") {
        scalar = "sin(" + x + ")";
        math_op = SHADER_GRAPH_MATH_SINE;
      } else if (op == "COSINE") {
        scalar = "cos(" + x + ")";
        math_op = SHADER_GRAPH_MATH_COSINE;
      } else if (op == "FRACT") {
        scalar = "frac(" + x + ")";
        math_op = SHADER_GRAPH_MATH_FRACT;
      } else if (op == "FLOOR") {
        scalar = "floor(" + x + ")";
        math_op = SHADER_GRAPH_MATH_FLOOR;
      } else {
        scalar = x + "+" + y;
      }
      expression = "(" + scalar + ").xxxx";
      slot = Emit(SHADER_GRAPH_OP_MATH, math_op, {a.slot, b.slot, c.slot});
    } else if (type == "color_ramp") {
      auto factor = Input(node, "factor", zero);
      if (!node.HasMember("elements") || !node["elements"].IsArray() || node["elements"].Empty()) {
        expression = factor.code;
        slot = Emit(SHADER_GRAPH_OP_COPY, 0, {factor.slot});
      } else {
        const auto &elements = node["elements"];
        expression = Literal(elements[0]["color"]).code;
        // Payload: (position, span, r, g, b, a) per element, where `span` is
        // the interval the HLSL bakes into the lerp for that element.
        std::vector<float> payload;
        auto append = [&payload](float position, float span, const glm::vec4 &color) {
          payload.insert(payload.end(), {position, span, color.x, color.y, color.z, color.w});
        };
        append(ReadNumber(elements[0]["position"]), 1.0f, LiteralValue(elements[0]["color"]));
        for (rapidjson::SizeType i = 1; i < elements.Size(); ++i) {
          double lo = ReadNumber(elements[i - 1]["position"]), hi = ReadNumber(elements[i]["position"]);
          auto t = "saturate(((" + factor.code + ").x-" + Number(lo) + ")/" + Number(std::max(hi - lo, 1e-8)) + ")";
          expression = "lerp(" + expression + "," + Literal(elements[i]["color"]).code + "," + t + ")";
          append(static_cast<float>(hi), static_cast<float>(std::max(hi - lo, 1e-8)),
                 LiteralValue(elements[i]["color"]));
        }
        slot = Emit(SHADER_GRAPH_OP_COLOR_RAMP, elements.Size(), {factor.slot}, PushData(payload),
                    static_cast<uint32_t>(payload.size()));
      }
    } else if (type == "rgb_curves") {
      auto color = Input(node, "color", zero), factor = Input(node, "factor", one);
      if (!node.HasMember("samples") || !node["samples"].IsArray() || node["samples"].Size() != 3)
        throw std::runtime_error("rgb_curves requires three sample arrays");
      std::array<std::string, 3> channels;
      const char *swizzle[] = {"x", "y", "z"};
      // Payload: one block per channel, `count, scale, count samples, count
      // sample positions`, holding the same constants the HLSL bakes in.
      std::vector<float> payload;
      for (int channel = 0; channel < 3; ++channel) {
        const auto &samples = node["samples"][channel];
        if (!samples.IsArray() || samples.Size() < 2)
          throw std::runtime_error("rgb_curves sample array is too short");
        std::string x = "saturate((" + color.code + ")." + swizzle[channel] + ")";
        std::string curve = Number(ReadNumber(samples[0]));
        for (rapidjson::SizeType i = 1; i < samples.Size(); ++i) {
          auto t = "saturate((" + x + "-" + Number(double(i - 1) / double(samples.Size() - 1)) + ")*" +
                   Number(double(samples.Size() - 1)) + ")";
          curve = "lerp(" + curve + "," + Number(ReadNumber(samples[i])) + "," + t + ")";
        }
        channels[channel] = curve;
        payload.push_back(static_cast<float>(samples.Size()));
        payload.push_back(static_cast<float>(double(samples.Size() - 1)));
        for (rapidjson::SizeType i = 0; i < samples.Size(); ++i)
          payload.push_back(ReadNumber(samples[i]));
        for (rapidjson::SizeType i = 0; i < samples.Size(); ++i)
          payload.push_back(static_cast<float>(double(i) / double(samples.Size() - 1)));
      }
      std::string curved =
          "float4(" + channels[0] + "," + channels[1] + "," + channels[2] + ",(" + color.code + ").w)";
      expression = "lerp(" + color.code + "," + curved + ",saturate((" + factor.code + ").x))";
      slot = Emit(SHADER_GRAPH_OP_RGB_CURVES, 0, {color.slot, factor.slot}, PushData(payload),
                  static_cast<uint32_t>(payload.size()));
    } else if (type == "hue_saturation") {
      auto color = Input(node, "color", zero), factor = Input(node, "factor", one);
      auto hue = Input(node, "hue", half);
      auto saturation = Input(node, "saturation", one), value = Input(node, "value", one);
      std::string hsv = "GraphRgbToHsv((" + color.code + ").xyz)";
      std::string adjusted = "GraphHsvToRgb(float3(frac((" + hsv + ").x+(" + hue.code + ").x-0.5f),max(0.0f,(" + hsv +
                             ").y*(" + saturation.code + ").x),(" + hsv + ").z*(" + value.code + ").x))";
      expression = "float4(lerp((" + color.code + ").xyz," + adjusted + ",saturate((" + factor.code + ").x)),(" +
                   color.code + ").w)";
      slot = Emit(SHADER_GRAPH_OP_HUE_SATURATION, 0,
                  {color.slot, factor.slot, hue.slot, saturation.slot, value.slot});
    } else if (type == "gamma") {
      auto color = Input(node, "color", one), gamma = Input(node, "gamma", one);
      expression =
          "float4(pow(max((" + color.code + ").xyz,0.0f),max((" + gamma.code + ").xxx,1e-6f)),(" + color.code + ").w)";
      slot = Emit(SHADER_GRAPH_OP_GAMMA, 0, {color.slot, gamma.slot});
    } else if (type == "bright_contrast") {
      auto color = Input(node, "color", one), brightness = Input(node, "brightness", zero),
           contrast = Input(node, "contrast", zero);
      expression = "float4((" + color.code + ").xyz*(1.0f+(" + contrast.code + ").x)+(" + brightness.code +
                   ").xxx,(" + color.code + ").w)";
      slot = Emit(SHADER_GRAPH_OP_BRIGHT_CONTRAST, 0, {color.slot, brightness.slot, contrast.slot});
    } else if (type == "layer_weight") {
      auto blend = Input(node, "blend", half);
      uint32_t sub = 0;
      if (output == "facing") {
        expression =
            "(1.0f-abs(dot(normalize(hit_record.normal),normalize("
            "view_direction)))).xxxx";
        sub = 1;
      } else {
        expression =
            "pow(1.0f-saturate(abs(dot(normalize(hit_record.normal),"
            "normalize(view_direction)))),max(0.01f,(" +
            blend.code + ").x*5.0f)).xxxx";
      }
      slot = Emit(SHADER_GRAPH_OP_LAYER_WEIGHT, sub, {blend.slot});
    } else if (type == "normal_map") {
      auto c = Input(node, "color", Const("float4(0.5,0.5,1,1)", {0.5f, 0.5f, 1.0f, 1.0f}));
      auto strength = Input(node, "strength", one);
      std::string tangent_normal =
          "normalize(lerp(float3(0,0,1),(" + c.code + ").xyz*2.0f-1.0f,saturate((" + strength.code + ").x)))";
      std::string mapped_normal = "normalize(mul(" + tangent_normal +
                                  ",float3x3(hit_record.tangent,cross(hit_record.normal,hit_record."
                                  "tangent)*hit_record.signal,hit_record.normal)))";
      expression = "float4(abs(hit_record.signal)>0.5f?" + mapped_normal + ":hit_record.normal,0)";
      slot = Emit(SHADER_GRAPH_OP_NORMAL_MAP, 0, {c.slot, strength.slot});
    } else if (type == "bump") {
      auto normal = InputLazy(node, "normal", [&] { return ShadingNormal(); });
      expression = normal.code;
      slot = Emit(SHADER_GRAPH_OP_COPY, 0, {normal.slot});
    } else if (type == "combine") {
      auto x = Input(node, "x", zero), y = Input(node, "y", zero), z = Input(node, "z", zero);
      expression = "float4((" + x.code + ").x,(" + y.code + ").x,(" + z.code + ").x,1.0f)";
      slot = Emit(SHADER_GRAPH_OP_COMBINE, 0, {x.slot, y.slot, z.slot});
    } else if (type == "separate") {
      auto c = Input(node, "color", zero);
      int component = (output == "green" || output == "y") ? 1 : (output == "blue" || output == "z") ? 2 : 0;
      const char *swizzle = component == 1 ? ".yyyy" : component == 2 ? ".zzzz" : ".xxxx";
      expression = "(" + c.code + ")" + swizzle;
      slot = Emit(SHADER_GRAPH_OP_SEPARATE, static_cast<uint32_t>(component), {c.slot});
    } else if (type == "passthrough") {
      auto value = Input(node, "value", zero);
      expression = value.code;
      slot = Emit(SHADER_GRAPH_OP_COPY, 0, {value.slot});
    } else if (type == "invert_y") {
      auto vector = Input(node, "vector", zero);
      expression = "float4((" + vector.code + ").x,1.0f-(" + vector.code + ").y,(" + vector.code + ").z,(" +
                   vector.code + ").w)";
      slot = Emit(SHADER_GRAPH_OP_INVERT_Y, 0, {vector.slot});
    } else if (type == "brick_texture") {
      auto p = InputLazy(node, "vector", [&] { return ObjectPosition(); });
      auto scale = Input(node, "scale", five);
      auto mortar_size = Input(node, "mortar_size", Const("float4(0.02,0.02,0.02,0.02)", {0.02f, 0.02f, 0.02f, 0.02f}));
      auto c1 = Input(node, "color1", zero), c2 = Input(node, "color2", one), mortar = Input(node, "mortar", zero);
      std::string q = "frac((" + p.code + ").xy*(" + scale.code + ").x)";
      std::string edge =
          "(min(min(" + q + ".x," + q + ".y),min(1.0f-" + q + ".x,1.0f-" + q + ".y))<(" + mortar_size.code + ").x)";
      expression = "(" + edge + "?" + mortar.code + ":lerp(" + c1.code + "," + c2.code + ",fmod(floor((" + p.code +
                   ").x*(" + scale.code + ").x)+floor((" + p.code + ").y*(" + scale.code + ").x),2.0f)))";
      slot = Emit(SHADER_GRAPH_OP_BRICK_TEXTURE, 0,
                  {p.slot, scale.slot, mortar_size.slot, c1.slot, c2.slot, mortar.slot});
    } else {
      throw std::runtime_error("unsupported shader node type: " + type);
    }
    lines_ << "  float4 " << var << "=" << expression << ";\n";
    active_.erase(key);
    const Expr result{var, slot};
    cache_[key] = result;
    return result;
  }

  const Value &graph_;
  const Value &nodes_;
  const std::map<std::string, int> &texture_slots_;
  std::ostringstream lines_;
  std::map<std::string, Expr> cache_;
  std::set<std::string> active_;
  int next_variable_ = 0;
  ShaderGraphProgram program_;
};
}  // namespace

std::unique_ptr<JsonScene> JsonScene::Load(Core *core, const std::filesystem::path &input_path, std::string *error) {
  try {
    auto path = std::filesystem::absolute(input_path).lexically_normal();
    std::ifstream stream(path);
    if (!stream)
      throw std::runtime_error("cannot open scene file: " + path.string());
    rapidjson::IStreamWrapper wrapper(stream);
    rapidjson::Document document;
    document.ParseStream<rapidjson::kParseCommentsFlag | rapidjson::kParseTrailingCommasFlag>(wrapper);
    if (document.HasParseError()) {
      std::ostringstream message;
      message << "JSON parse error at byte " << document.GetErrorOffset() << ": "
              << rapidjson::GetParseError_En(document.GetParseError());
      throw std::runtime_error(message.str());
    }
    if (!document.IsObject())
      throw std::runtime_error("scene root must be an object");
    if (!document.HasMember("format") || ReadString(document["format"]) != "sparkium-scene")
      throw std::runtime_error("unsupported or missing format (expected 'sparkium-scene')");
    if (!document.HasMember("version") || ReadInt(document["version"]) != 1)
      throw std::runtime_error("unsupported scene version (expected 1)");

    auto result = std::unique_ptr<JsonScene>(new JsonScene);
    result->core_ = core;
    result->path_ = path;
    result->name_ = document.HasMember("name") ? ReadString(document["name"]) : path.stem().string();
    result->scene_ = std::make_unique<Scene>(core);

    const auto &renderer = RequireObject(Member(document, "renderer"));
    result->scene_->settings.raytracing.samples_per_dispatch =
        renderer.HasMember("samples_per_dispatch") ? ReadInt(renderer["samples_per_dispatch"]) : 32;
    result->scene_->settings.raytracing.max_bounces =
        renderer.HasMember("max_bounces") ? ReadInt(renderer["max_bounces"]) : 32;
    if (result->scene_->settings.raytracing.samples_per_dispatch <= 0 ||
        result->scene_->settings.raytracing.max_bounces <= 0)
      throw std::runtime_error("samples_per_dispatch and max_bounces must be positive");
    result->scene_->settings.raytracing.alpha_shadow = BoolMember(renderer, "alpha_shadow", false);
    result->scene_->settings.raster.ambient_light = Vec3Member(renderer, "ambient_light", {0.1f, 0.1f, 0.1f});
    result->scene_->settings.raytracing.background_color =
        Vec3Member(renderer, "background_color", result->scene_->settings.raster.ambient_light);
    std::string pipeline = renderer.HasMember("pipeline") ? ReadString(renderer["pipeline"]) : "auto";
    if (pipeline == "rasterization")
      result->render_pipeline_ = RENDER_PIPELINE_RASTERIZATION;
    else if (pipeline == "ray_tracing")
      result->render_pipeline_ = RENDER_PIPELINE_RAY_TRACING;
    else if (pipeline == "ray_query")
      result->render_pipeline_ = RENDER_PIPELINE_RAY_QUERY;
    else if (pipeline == "rt_fallback")
      result->render_pipeline_ = RENDER_PIPELINE_RT_FALLBACK;
    else if (pipeline == "auto")
      result->render_pipeline_ = RENDER_PIPELINE_AUTO;
    else
      throw std::runtime_error("renderer.pipeline must be auto, rasterization, ray_tracing, rt_fallback, or ray_query");

    const auto &film = RequireObject(Member(document, "film"));
    int width = ReadInt(Member(film, "width"));
    int height = ReadInt(Member(film, "height"));
    if (width <= 0 || height <= 0)
      throw std::runtime_error("film dimensions must be positive");
    result->film_ = std::make_unique<Film>(core, width, height);
    result->film_->info.persistence = FloatMember(film, "persistence", 1.0f);
    result->film_->info.clamping = FloatMember(film, "clamping", 100.0f);
    result->film_->info.max_exposure = FloatMember(film, "max_exposure", 1.0f);
    std::string view_transform = film.HasMember("view_transform") ? ReadString(film["view_transform"]) : "normalized";
    if (view_transform == "normalized")
      result->film_->info.view_transform = 0;
    else if (view_transform == "standard")
      result->film_->info.view_transform = 1;
    else if (view_transform == "filmic")
      result->film_->info.view_transform = 2;
    else
      throw std::runtime_error("film.view_transform must be normalized, standard, or filmic");
    result->film_->info.exposure = FloatMember(film, "exposure", 0.0f);
    result->film_->info.gamma = FloatMember(film, "gamma", 1.0f);
    result->film_->info.contrast = FloatMember(film, "contrast", 1.0f);

    const auto &camera = RequireObject(Member(document, "camera"));
    auto eye = Vec3Member(camera, "eye");
    auto target = Vec3Member(camera, "target");
    auto up = Vec3Member(camera, "up", {0.0f, 1.0f, 0.0f});
    float fov = FloatMember(camera, "fov_degrees", 60.0f);
    if (!(fov > 0.0f && fov < 180.0f) || glm::length(target - eye) < 1e-6f ||
        glm::length(glm::cross(target - eye, up)) < 1e-6f)
      throw std::runtime_error("camera requires a valid field of view and nondegenerate look-at vectors");
    result->camera_ = std::make_unique<Camera>(core, glm::lookAt(eye, target, up), glm::radians(fov),
                                               static_cast<float>(width) / static_cast<float>(height));
    result->camera_->aperture_radius = FloatMember(camera, "aperture_radius", 0.0f);
    result->camera_->focus_distance = FloatMember(camera, "focus_distance", glm::length(target - eye));
    result->camera_->aperture_blades = camera.HasMember("aperture_blades") ? ReadInt(camera["aperture_blades"]) : 0;
    result->camera_->aperture_rotation = FloatMember(camera, "aperture_rotation", 0.0f);
    result->camera_->aperture_ratio = FloatMember(camera, "aperture_ratio", 1.0f);

    const auto &materials = RequireObject(Member(document, "materials"));
    for (auto it = materials.MemberBegin(); it != materials.MemberEnd(); ++it) {
      std::string id = ReadString(it->name);
      const auto &spec = RequireObject(it->value);
      std::string type = ReadString(Member(spec, "type"));
      std::unique_ptr<Material> material;
      if (type == "lambertian") {
        material = std::make_unique<MaterialLambertian>(core, Vec3Member(spec, "base_color", {0.8f, 0.8f, 0.8f}),
                                                        Vec3Member(spec, "emission"));
      } else if (type == "specular") {
        material = std::make_unique<MaterialSpecular>(core, Vec3Member(spec, "base_color", {0.8f, 0.8f, 0.8f}));
      } else if (type == "light") {
        material = std::make_unique<MaterialLight>(
            core, Vec3Member(spec, "emission"), BoolMember(spec, "two_sided", false),
            BoolMember(spec, "block_ray", false), BoolMember(spec, "camera_visible", true),
            FloatMember(spec, "falloff_distance", 0.0f));
      } else if (type == "principled") {
        auto principled = std::make_unique<MaterialPrincipled>(core, Vec3Member(spec, "base_color", glm::vec3{0.8f}));
        principled->subsurface_color = Vec3Member(spec, "subsurface_color", principled->subsurface_color);
        principled->subsurface_radius = Vec3Member(spec, "subsurface_radius", principled->subsurface_radius);
#define READ_FLOAT(field) principled->field = FloatMember(spec, #field, principled->field)
        READ_FLOAT(subsurface);
        READ_FLOAT(metallic);
        READ_FLOAT(specular);
        READ_FLOAT(specular_tint);
        READ_FLOAT(roughness);
        READ_FLOAT(anisotropic);
        READ_FLOAT(anisotropic_rotation);
        READ_FLOAT(sheen);
        READ_FLOAT(sheen_tint);
        READ_FLOAT(clearcoat);
        READ_FLOAT(clearcoat_roughness);
        READ_FLOAT(ior);
        READ_FLOAT(transmission);
        READ_FLOAT(transmission_roughness);
        READ_FLOAT(emission_strength);
#undef READ_FLOAT
        principled->emission_color = Vec3Member(spec, "emission_color", principled->emission_color);
        if (spec.HasMember("textures")) {
          const auto &textures = RequireObject(spec["textures"]);
          const std::pair<const char *, graphics::Image **> slots[] = {
              {"normal", &principled->textures.normal},
              {"base_color", &principled->textures.base_color},
              {"metallic", &principled->textures.metallic},
              {"specular", &principled->textures.specular},
              {"roughness", &principled->textures.roughness},
              {"anisotropic", &principled->textures.anisotropic},
              {"anisotropic_rotation", &principled->textures.anisotropic_rotation},
              {"emission", &principled->textures.emission}};
          for (auto [slot, destination] : slots) {
            if (!textures.HasMember(slot))
              continue;
            auto image = std::unique_ptr<graphics::Image>{};
            auto asset = Resolve(path, ReadString(textures[slot]));
            if (graphics::LoadImageFromFile(core->GraphicsCore(), asset.string(), &image) != 0)
              throw std::runtime_error("cannot load texture: " + asset.string());
            *destination = image.get();
            result->images_.push_back(std::move(image));
          }
          principled->textures.normal_reverse_y = BoolMember(textures, "normal_reverse_y", false);
        }
        material = std::move(principled);
      } else if (type == "shader_graph") {
        const auto &graph = RequireObject(Member(spec, "graph"));
        const auto &nodes = RequireObject(Member(graph, "nodes"));
        std::map<std::string, int> texture_slots;
        std::vector<graphics::Image *> textures;
        for (auto node = nodes.MemberBegin(); node != nodes.MemberEnd(); ++node) {
          if (!RequireObject(node->value).HasMember("type") ||
              std::string(ReadString(node->value["type"])) != "image_texture")
            continue;
          auto asset = Resolve(path, ReadString(Member(node->value, "path")));
          auto image = std::unique_ptr<graphics::Image>{};
          if (graphics::LoadImageFromFile(core->GraphicsCore(), asset.string(), &image) != 0)
            throw std::runtime_error("cannot load shader graph texture: " + asset.string());
          texture_slots[ReadString(node->name)] = static_cast<int>(textures.size());
          textures.push_back(image.get());
          result->images_.push_back(std::move(image));
        }
        ShaderGraphCompiler compiler(graph, texture_slots);
        auto code = compiler.Compile();
        material = std::make_unique<MaterialShaderGraph>(core, code, textures, Vec3Member(spec, "emission_hint"),
                                                         compiler.Program());
      } else {
        throw std::runtime_error("unknown material type: " + type);
      }
      result->materials_.emplace(id, std::move(material));
    }

    const auto &geometries = RequireObject(Member(document, "geometries"));
    for (auto it = geometries.MemberBegin(); it != geometries.MemberEnd(); ++it) {
      std::string id = ReadString(it->name);
      const auto &spec = RequireObject(it->value);
      std::string type = ReadString(Member(spec, "type"));
      if (type == "hair") {
        auto hair = BinaryHair(Resolve(path, ReadString(Member(spec, "path"))));
        int radial_segments = spec.HasMember("radial_segments") ? ReadInt(spec["radial_segments"]) : 3;
        result->geometries_.emplace(
            id, std::make_unique<GeometryHair>(core, hair.points, hair.radii, hair.offsets, radial_segments));
        continue;
      }
      Mesh<> mesh;
      if (type == "sphere") {
        int longitude = spec.HasMember("longitude_segments") ? ReadInt(spec["longitude_segments"]) : 30;
        int latitude = spec.HasMember("latitude_segments") ? ReadInt(spec["latitude_segments"]) : -1;
        if (longitude < 3 || (latitude != -1 && latitude < 2))
          throw std::runtime_error("sphere requires at least 3 longitude and 2 latitude segments");
        mesh = Mesh<>::Sphere(longitude, latitude);
      } else if (type == "mesh") {
        auto asset = Resolve(path, ReadString(Member(spec, "path")));
        if (mesh.LoadObjFile(asset.string()) != 0)
          throw std::runtime_error("cannot load mesh: " + asset.string());
      } else if (type == "binary_mesh") {
        mesh = BinaryMesh(Resolve(path, ReadString(Member(spec, "path"))));
      } else if (type == "inline_mesh") {
        mesh = InlineMesh(spec);
      } else {
        throw std::runtime_error("unknown geometry type: " + type);
      }
      if (BoolMember(spec, "generate_normals", false))
        mesh.GenerateNormals();
      if (BoolMember(spec, "generate_tangents", false))
        mesh.GenerateTangents();
      result->geometries_.emplace(id, std::make_unique<GeometryMesh>(core, mesh));
    }

    const auto &entities = RequireArray(Member(document, "entities"));
    for (const auto &spec : entities.GetArray()) {
      std::string type = ReadString(Member(spec, "type"));
      Entity *entity_ptr = nullptr;
      if (type == "mesh") {
        std::string geometry_id = ReadString(Member(spec, "geometry"));
        std::string material_id = ReadString(Member(spec, "material"));
        auto geometry = result->geometries_.find(geometry_id);
        auto material = result->materials_.find(material_id);
        if (geometry == result->geometries_.end())
          throw std::runtime_error("unknown geometry: " + geometry_id);
        if (material == result->materials_.end())
          throw std::runtime_error("unknown material: " + material_id);
        glm::mat4 transform{1.0f};
        if (spec.HasMember("transform"))
          transform = Transform(spec["transform"]);
        if (spec.HasMember("look_at")) {
          const auto &look = RequireObject(spec["look_at"]);
          auto position = Vec3Member(look, "position");
          auto target = Vec3Member(look, "target");
          auto up = Vec3Member(look, "up", {0.0f, 1.0f, 0.0f});
          auto scale = Vec3Member(look, "scale", glm::vec3{1.0f});
          transform = glm::inverse(glm::lookAt(position, target, up)) * glm::scale(glm::mat4{1.0f}, scale);
        }
        auto entity =
            std::make_unique<EntityGeometryMaterial>(core, geometry->second.get(), material->second.get(), transform);
        entity->raster_light = BoolMember(spec, "raster_light", true);
        entity_ptr = entity.get();
        result->entities_.push_back(std::move(entity));
      } else if (type == "point_light") {
        auto entity = std::make_unique<EntityPointLight>(
            core, Vec3Member(spec, "position"), Vec3Member(spec, "color", glm::vec3{1.0f}),
            FloatMember(spec, "strength", 0.0f), FloatMember(spec, "radius", 0.0f),
            BoolMember(spec, "soft_falloff", false), FloatMember(spec, "sampling_weight", -1.0f));
        entity_ptr = entity.get();
        result->entities_.push_back(std::move(entity));
      } else {
        throw std::runtime_error("unknown entity type: " + type);
      }
      result->scene_->AddEntity(entity_ptr);
    }
    return result;
  } catch (const std::exception &exception) {
    if (error)
      *error = exception.what();
    return nullptr;
  }
}

std::vector<std::filesystem::path> FindJsonScenes(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> result;
  if (!std::filesystem::exists(directory))
    return result;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().filename() == "scene.json")
      result.push_back(entry.path());
  }
  std::sort(result.begin(), result.end());
  return result;
}
}  // namespace sparkium
