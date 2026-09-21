#include "sparkium/backend/cuda/shader_graph.h"

#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

#include "rapidjson/document.h"

namespace sparkium::backend::cuda::detail {
namespace {
using Value = rapidjson::Value;

const Value &RequireObject(const Value &value) {
  if (!value.IsObject())
    throw std::runtime_error("expected a JSON object");
  return value;
}

std::string ReadString(const Value &value) {
  if (!value.IsString())
    throw std::runtime_error("expected a string");
  return value.GetString();
}

float ReadNumber(const Value &value) {
  if (!value.IsNumber())
    throw std::runtime_error("expected a number");
  float result = value.GetFloat();
  if (!std::isfinite(result))
    throw std::runtime_error("expected a finite float");
  return result;
}

const Value &Member(const Value &value, const char *name) {
  if (!value.IsObject() || !value.HasMember(name))
    throw std::runtime_error(std::string("missing field '") + name + "'");
  return value[name];
}

class ShaderGraphCompiler {
 public:
  ShaderGraphCompiler(const Value &graph, const std::map<std::string, int> &texture_slots)
      : graph_(graph),
        nodes_(Member(graph, "nodes")),
        texture_slots_(texture_slots) {
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
GraphSurface EvaluateShaderGraph(SP_CONTEXT HitRecord hit_record, float3 view_direction, int bounce, int ray_type, bool is_shadow_ray,
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
    for (const auto &[name, target] : outputs) {
      if (!surface.HasMember(name))
        continue;
      auto expression = Expression(surface[name]);
      assignments.emplace_back(target,
                               expression + ((std::string(name) == "base_color" || std::string(name) == "emission" ||
                                              std::string(name) == "normal" || std::string(name) == "subsurface_radius")
                                                 ? ".xyz"
                                                 : ".x"));
    }
    for (const auto &[target, expression] : assignments)
      lines_ << "  " << target << "=" << expression << ";\n";
    lines_ << "  return surface;\n}\n";
    return CodeLines(lines_.str());
  }

 private:
  std::string Number(double value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    auto text = out.str();
    if (text.find_first_of(".eE") == std::string::npos)
      text += ".0";
    return text + "f";
  }

  std::string Literal(const Value &value) {
    if (value.IsNumber()) {
      auto n = Number(ReadNumber(value));
      return "float4(" + n + "," + n + "," + n + "," + n + ")";
    }
    if (!value.IsArray() || value.Empty() || value.Size() > 4)
      throw std::runtime_error("shader input must be a number, vector, or node reference");
    std::vector<std::string> parts;
    for (const auto &item : value.GetArray())
      parts.push_back(Number(ReadNumber(item)));
    while (parts.size() < 4)
      parts.push_back(parts.size() == 3 ? "1.0f" : parts.back());
    return "float4(" + parts[0] + "," + parts[1] + "," + parts[2] + "," + parts[3] + ")";
  }

  std::string Input(const Value &node, const char *name, const std::string &fallback) {
    if (!node.HasMember("inputs") || !RequireObject(node["inputs"]).HasMember(name))
      return fallback;
    return Expression(node["inputs"][name]);
  }

  std::string Expression(const Value &value) {
    if (!value.IsObject() || !value.HasMember("node"))
      return Literal(value);
    return Node(ReadString(value["node"]), value.HasMember("output") ? ReadString(value["output"]) : "value");
  }

  std::string Node(const std::string &id, const std::string &output) {
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
    const auto zero = "float4(0,0,0,0)", one = "float4(1,1,1,1)";
    if (type == "value" || type == "rgb") {
      expression = Input(node, "value", zero);
    } else if (type == "texture_coordinate") {
      if (output == "uv")
        expression = "float4(hit_record.tex_coord,0,0)";
      else if (output == "normal")
        expression = "float4(hit_record.normal,0)";
      else
        expression = "float4(hit_record.object_position,0)";
    } else if (type == "image_texture") {
      auto it = texture_slots_.find(id);
      if (it == texture_slots_.end())
        throw std::runtime_error("image node has no loaded texture: " + id);
      auto vector = Input(node, "vector", "float4(hit_record.tex_coord,0,0)");
      expression = "SampleTexture(SP_CONTEXT_ARG material_data.Load(" + std::to_string(12 + it->second * 4) + "),(" +
                   vector + ").xy)";
      if (output == "alpha")
        expression = "(" + expression + ").wwww";
      else if (node.HasMember("color_space") && std::string(ReadString(node["color_space"])) == "srgb")
        expression = "float4(GraphSrgbToLinear((" + expression + ").xyz),(" + expression + ").w)";
    } else if (type == "mapping") {
      auto vector = Input(node, "vector", zero), location = Input(node, "location", zero);
      auto rotation = Input(node, "rotation", zero), scale = Input(node, "scale", one);
      expression = "float4(GraphRotateXYZ((" + vector + ").xyz*(" + scale + ").xyz,(" + rotation + ").xyz)+(" +
                   location + ").xyz,0)";
    } else if (type == "noise_texture") {
      auto vector = Input(node, "vector", "float4(hit_record.object_position,0)");
      auto scale = Input(node, "scale", "float4(5,5,5,5)");
      expression = "GraphNoise((" + vector + ").xyz*(" + scale + ").x).xxxx";
    } else if (type == "voronoi_texture") {
      auto vector = Input(node, "vector", "float4(hit_record.object_position,0)");
      auto scale = Input(node, "scale", "float4(5,5,5,5)");
      expression = "GraphVoronoi((" + vector + ").xyz*(" + scale + ").x).xxxx";
    } else if (type == "gradient_texture") {
      auto vector = Input(node, "vector", "float4(hit_record.object_position,0)");
      expression = "(" + vector + ").xxxx";
    } else if (type == "wave_texture") {
      auto vector = Input(node, "vector", "float4(hit_record.object_position,0)");
      auto scale = Input(node, "scale", "float4(5,5,5,5)");
      auto distortion = Input(node, "distortion", zero), phase = Input(node, "phase", zero);
      std::string wave_type = node.HasMember("wave_type") ? ReadString(node["wave_type"]) : "BANDS";
      std::string direction = node.HasMember("bands_direction") ? ReadString(node["bands_direction"]) : "X";
      std::string p = "(" + vector + ").xyz*(" + scale + ").x";
      std::string coordinate;
      if (wave_type == "RINGS")
        coordinate = "length(" + p + ")";
      else if (direction == "Y")
        coordinate = "(" + p + ").y";
      else if (direction == "Z")
        coordinate = "(" + p + ").z";
      else if (direction == "DIAGONAL")
        coordinate = "((" + p + ").x+(" + p + ").y+(" + p + ").z)*0.577350269f";
      else
        coordinate = "(" + p + ").x";
      coordinate = "(" + coordinate + "+(" + distortion + ").x*GraphNoise(" + p + ")+(" + phase + ").x)";
      expression = "(0.5f+0.5f*sin(" + coordinate + "*6.283185307f)).xxxx";
    } else if (type == "sky_texture") {
      auto vector = Input(node, "vector", "float4(0,0,1,0)");
      auto sun = node.HasMember("sun_direction") ? Literal(node["sun_direction"]) : "float4(0,1,0,0)";
      expression =
          "float4(lerp(float3(0.08f,0.16f,0.35f),float3(0.65f,0.78f,1."
          "0f),saturate(normalize((" +
          vector + ").xyz).z*0.5f+0.5f))+pow(saturate(dot(normalize((" + vector + ").xyz),normalize((" + sun +
          ").xyz))),512.0f)*float3(8,6,3),1)";
    } else if (type == "vertex_attribute") {
      if (output == "alpha")
        expression = "float4(1,1,1,1)";
      else if (output == "factor")
        expression = "hit_record.color.xxxx";
      else
        expression = "float4(hit_record.color,1)";
    } else if (type == "object_info") {
      if (output == "location")
        expression = "float4(hit_record.object_origin,0)";
      else if (output == "object_index" || output == "material_index")
        expression = "float(hit_record.object_index).xxxx";
      else if (output == "random")
        expression = "GraphHash(float3(hit_record.object_index,17,31)).xxxx";
      else
        expression = "float4(1,1,1,1)";
    } else if (type == "geometry_info") {
      if (output == "position")
        expression = "float4(hit_record.position,1)";
      else if (output == "normal" || output == "true_normal")
        expression = "float4(hit_record.normal,0)";
      else if (output == "incoming")
        expression = "float4(-view_direction,0)";
      else if (output == "backfacing")
        expression = "(hit_record.front_facing?0.0f:1.0f).xxxx";
      else
        expression = zero;
    } else if (type == "light_path") {
      if (output == "is_camera_ray")
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_CAMERA?1.0f:0.0f).xxxx";
      else if (output == "is_shadow_ray")
        expression = "(is_shadow_ray?1.0f:0.0f).xxxx";
      else if (output == "is_reflection_ray")
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_REFLECTION?1.0f:0.0f).xxxx";
      else if (output == "is_transmission_ray")
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_TRANSMISSION?1.0f:0.0f).xxxx";
      else if (output == "is_glossy_ray" || output == "is_diffuse_ray" || output == "is_singular_ray")
        expression = "(!is_shadow_ray&&ray_type==RAY_TYPE_REFLECTION?1.0f:0.0f).xxxx";
      else if (output == "ray_length")
        expression = "hit_record.t.xxxx";
      else
        expression = zero;
    } else if (type == "invert") {
      expression = "lerp(" + Input(node, "color", zero) + ",1.0f-" + Input(node, "color", zero) + ",(" +
                   Input(node, "factor", one) + ").x)";
    } else if (type == "mix") {
      auto factor = Input(node, "factor", "float4(0.5,0.5,0.5,0.5)");
      auto a = Input(node, "a", zero), b = Input(node, "b", one);
      std::string blend = node.HasMember("blend_type") ? ReadString(node["blend_type"]) : "MIX";
      if (blend == "MULTIPLY")
        expression = "lerp(" + a + "," + a + "*" + b + ",saturate((" + factor + ").x))";
      else if (blend == "ADD")
        expression = a + "+" + b + "*(" + factor + ").x";
      else if (blend == "SUBTRACT")
        expression = a + "-" + b + "*(" + factor + ").x";
      else if (blend == "DARKEN")
        expression = "lerp(" + a + ",min(" + a + "," + b + "),saturate((" + factor + ").x))";
      else if (blend == "LIGHTEN")
        expression = "lerp(" + a + ",max(" + a + "," + b + "),saturate((" + factor + ").x))";
      else if (blend == "SCREEN")
        expression = "lerp(" + a + ",1.0f-(1.0f-" + a + ")*(1.0f-" + b + "),saturate((" + factor + ").x))";
      else if (blend == "DIFFERENCE")
        expression = "lerp(" + a + ",abs(" + a + "-" + b + "),saturate((" + factor + ").x))";
      else if (blend == "DIVIDE")
        expression = "lerp(" + a + "," + a + "/max(abs(" + b + "),1e-8f),saturate((" + factor + ").x))";
      else if (blend == "OVERLAY") {
        auto overlay =
            "lerp(2.0f*" + a + "*" + b + ",1.0f-2.0f*(1.0f-" + a + ")*(1.0f-" + b + "),step(0.5f," + a + "))";
        expression = "lerp(" + a + "," + overlay + ",saturate((" + factor + ").x))";
      } else
        expression = "lerp(" + a + "," + b + ",saturate((" + factor + ").x))";
    } else if (type == "math") {
      auto a = Input(node, "a", zero), b = Input(node, "b", zero), c = Input(node, "c", zero);
      std::string op = node.HasMember("operation") ? ReadString(node["operation"]) : "ADD";
      std::string x = "(" + a + ").x", y = "(" + b + ").x", z = "(" + c + ").x", scalar;
      if (op == "MULTIPLY")
        scalar = x + "*" + y;
      else if (op == "SUBTRACT")
        scalar = x + "-" + y;
      else if (op == "DIVIDE")
        scalar = x + "/max(abs(" + y + "),1e-8f)";
      else if (op == "POWER")
        scalar = "pow(max(" + x + ",0.0f)," + y + ")";
      else if (op == "MINIMUM")
        scalar = "min(" + x + "," + y + ")";
      else if (op == "MAXIMUM")
        scalar = "max(" + x + "," + y + ")";
      else if (op == "LESS_THAN")
        scalar = "(" + x + "<" + y + "?1.0f:0.0f)";
      else if (op == "GREATER_THAN")
        scalar = "(" + x + ">" + y + "?1.0f:0.0f)";
      else if (op == "MULTIPLY_ADD")
        scalar = x + "*" + y + "+" + z;
      else if (op == "ABSOLUTE")
        scalar = "abs(" + x + ")";
      else if (op == "SINE")
        scalar = "sin(" + x + ")";
      else if (op == "COSINE")
        scalar = "cos(" + x + ")";
      else if (op == "FRACT")
        scalar = "frac(" + x + ")";
      else if (op == "FLOOR")
        scalar = "floor(" + x + ")";
      else
        scalar = x + "+" + y;
      expression = "(" + scalar + ").xxxx";
    } else if (type == "color_ramp") {
      auto factor = Input(node, "factor", zero);
      if (!node.HasMember("elements") || !node["elements"].IsArray() || node["elements"].Empty())
        expression = factor;
      else {
        const auto &elements = node["elements"];
        expression = Literal(elements[0]["color"]);
        for (rapidjson::SizeType i = 1; i < elements.Size(); ++i) {
          double lo = ReadNumber(elements[i - 1]["position"]), hi = ReadNumber(elements[i]["position"]);
          auto t = "saturate(((" + factor + ").x-" + Number(lo) + ")/" + Number(std::max(hi - lo, 1e-8)) + ")";
          expression = "lerp(" + expression + "," + Literal(elements[i]["color"]) + "," + t + ")";
        }
      }
    } else if (type == "rgb_curves") {
      auto color = Input(node, "color", zero), factor = Input(node, "factor", one);
      if (!node.HasMember("samples") || !node["samples"].IsArray() || node["samples"].Size() != 3)
        throw std::runtime_error("rgb_curves requires three sample arrays");
      std::array<std::string, 3> channels;
      const char *swizzle[] = {"x", "y", "z"};
      for (int channel = 0; channel < 3; ++channel) {
        const auto &samples = node["samples"][channel];
        if (!samples.IsArray() || samples.Size() < 2)
          throw std::runtime_error("rgb_curves sample array is too short");
        std::string x = "saturate((" + color + ")." + swizzle[channel] + ")";
        std::string curve = Number(ReadNumber(samples[0]));
        for (rapidjson::SizeType i = 1; i < samples.Size(); ++i) {
          auto t = "saturate((" + x + "-" + Number(double(i - 1) / double(samples.Size() - 1)) + ")*" +
                   Number(double(samples.Size() - 1)) + ")";
          curve = "lerp(" + curve + "," + Number(ReadNumber(samples[i])) + "," + t + ")";
        }
        channels[channel] = curve;
      }
      std::string curved = "float4(" + channels[0] + "," + channels[1] + "," + channels[2] + ",(" + color + ").w)";
      expression = "lerp(" + color + "," + curved + ",saturate((" + factor + ").x))";
    } else if (type == "hue_saturation") {
      auto color = Input(node, "color", zero), factor = Input(node, "factor", one);
      auto hue = Input(node, "hue", "float4(0.5,0.5,0.5,0.5)");
      auto saturation = Input(node, "saturation", one), value = Input(node, "value", one);
      std::string hsv = "GraphRgbToHsv((" + color + ").xyz)";
      std::string adjusted = "GraphHsvToRgb(float3(frac((" + hsv + ").x+(" + hue + ").x-0.5f),max(0.0f,(" + hsv +
                             ").y*(" + saturation + ").x),(" + hsv + ").z*(" + value + ").x))";
      expression = "float4(lerp((" + color + ").xyz," + adjusted + ",saturate((" + factor + ").x)),(" + color + ").w)";
    } else if (type == "gamma") {
      auto color = Input(node, "color", one), gamma = Input(node, "gamma", one);
      expression = "float4(pow(max((" + color + ").xyz,0.0f),max((" + gamma + ").xxx,1e-6f)),(" + color + ").w)";
    } else if (type == "bright_contrast") {
      auto color = Input(node, "color", one), brightness = Input(node, "brightness", zero),
           contrast = Input(node, "contrast", zero);
      expression = "float4((" + color + ").xyz*(1.0f+(" + contrast + ").x)+(" + brightness + ").xxx,(" + color + ").w)";
    } else if (type == "layer_weight") {
      auto blend = Input(node, "blend", "float4(0.5,0.5,0.5,0.5)");
      if (output == "facing")
        expression =
            "(1.0f-abs(dot(normalize(hit_record.normal),normalize("
            "view_direction)))).xxxx";
      else
        expression =
            "pow(1.0f-saturate(abs(dot(normalize(hit_record.normal),"
            "normalize(view_direction)))),max(0.01f,(" +
            blend + ").x*5.0f)).xxxx";
    } else if (type == "normal_map") {
      auto c = Input(node, "color", "float4(0.5,0.5,1,1)");
      auto strength = Input(node, "strength", one);
      std::string tangent_normal =
          "normalize(lerp(float3(0,0,1),(" + c + ").xyz*2.0f-1.0f,saturate((" + strength + ").x)))";
      std::string mapped_normal = "normalize(mul(" + tangent_normal +
                                  ",float3x3(hit_record.tangent,cross(hit_record.normal,hit_record."
                                  "tangent)*hit_record.signal,hit_record.normal)))";
      expression = "float4(abs(hit_record.signal)>0.5f?" + mapped_normal + ":hit_record.normal,0)";
    } else if (type == "bump") {
      expression = Input(node, "normal", "float4(hit_record.normal,0)");
    } else if (type == "combine") {
      expression = "float4((" + Input(node, "x", zero) + ").x,(" + Input(node, "y", zero) + ").x,(" +
                   Input(node, "z", zero) + ").x,1.0f)";
    } else if (type == "separate") {
      auto c = Input(node, "color", zero);
      int component = (output == "green" || output == "y") ? 1 : (output == "blue" || output == "z") ? 2 : 0;
      const char *swizzle = component == 1 ? ".yyyy" : component == 2 ? ".zzzz" : ".xxxx";
      expression = "(" + c + ")" + swizzle;
    } else if (type == "passthrough") {
      expression = Input(node, "value", zero);
    } else if (type == "invert_y") {
      auto vector = Input(node, "vector", zero);
      expression = "float4((" + vector + ").x,1.0f-(" + vector + ").y,(" + vector + ").z,(" + vector + ").w)";
    } else if (type == "brick_texture") {
      auto p = Input(node, "vector", "float4(hit_record.object_position,0)");
      auto scale = Input(node, "scale", "float4(5,5,5,5)");
      auto mortar_size = Input(node, "mortar_size", "float4(0.02,0.02,0.02,0.02)");
      auto c1 = Input(node, "color1", zero), c2 = Input(node, "color2", one), mortar = Input(node, "mortar", zero);
      std::string q = "frac((" + p + ").xy*(" + scale + ").x)";
      std::string edge =
          "(min(min(" + q + ".x," + q + ".y),min(1.0f-" + q + ".x,1.0f-" + q + ".y))<(" + mortar_size + ").x)";
      expression = "(" + edge + "?" + mortar + ":lerp(" + c1 + "," + c2 + ",fmod(floor((" + p + ").x*(" + scale +
                   ").x)+floor((" + p + ").y*(" + scale + ").x),2.0f)))";
    } else {
      throw std::runtime_error("unsupported shader node type: " + type);
    }
    lines_ << "  float4 " << var << "=" << expression << ";\n";
    active_.erase(key);
    cache_[key] = var;
    return var;
  }

  const Value &graph_;
  const Value &nodes_;
  const std::map<std::string, int> &texture_slots_;
  std::ostringstream lines_;
  std::map<std::string, std::string> cache_;
  std::set<std::string> active_;
  int next_variable_ = 0;
};

rapidjson::Value Convert(const NodeValue &value, rapidjson::Document::AllocatorType &allocator) {
  return std::visit(
      [&](const auto &v) -> rapidjson::Value {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>)
          return {};
        else if constexpr (std::is_same_v<T, bool>)
          return rapidjson::Value(v);
        else if constexpr (std::is_same_v<T, double>)
          return rapidjson::Value(v);
        else if constexpr (std::is_same_v<T, std::string>)
          return rapidjson::Value(v.c_str(), allocator);
        else if constexpr (std::is_same_v<T, NodeValue::Array>) {
          rapidjson::Value array(rapidjson::kArrayType);
          for (const auto &item : v)
            array.PushBack(Convert(item, allocator), allocator);
          return array;
        } else {
          rapidjson::Value object(rapidjson::kObjectType);
          for (const auto &[key, item] : v)
            object.AddMember(rapidjson::Value(key.c_str(), allocator), Convert(item, allocator), allocator);
          return object;
        }
      },
      value.value);
}
}  // namespace

CodeLines CompileShaderGraph(const NodeValue &graph, const std::map<std::string, int> &textures) {
  rapidjson::Document document;
  auto value = Convert(graph, document.GetAllocator());
  return ShaderGraphCompiler(value, textures).Compile();
}
}  // namespace sparkium::backend::cuda::detail
