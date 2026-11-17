#include "aero_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/istreamwrapper.h"

namespace wind3d::aero {
namespace {
Vector Add(Vector a, Vector b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector Scale(Vector a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

Vector Cross(Vector a, Vector b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Transform Compose(const Transform &parent, const Transform &child) {
  Transform result;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      result.rotation[i * 3 + j] = 0;
      for (int k = 0; k < 3; ++k)
        result.rotation[i * 3 + j] += parent.rotation[i * 3 + k] * child.rotation[k * 3 + j];
    }
  result.translation = parent.Point(child.translation);
  return result;
}

Transform Hinge(Vector axis, Vector pivot, double angle) {
  double radians = angle * 0.017453292519943295, c = std::cos(radians), s = std::sin(radians), t = 1 - c;
  double x = axis.x, y = axis.y, z = axis.z;
  Transform result;
  result.rotation = {t * x * x + c,     t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c,
                     t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c};
  result.translation = Add(pivot, Scale(result.Rotate(pivot), -1));
  return result;
}

using Json = rapidjson::Value;

bool Number(const Json &v, double &out) {
  if (!v.IsNumber() || !std::isfinite(v.GetDouble()))
    return false;
  out = v.GetDouble();
  return true;
}

bool VectorValue(const Json &v, Vector &out) {
  return v.IsArray() && v.Size() == 3 && Number(v[0], out.x) && Number(v[1], out.y) && Number(v[2], out.z);
}

bool Text(const Json &v, std::string &out) {
  if (!v.IsString() || !v.GetStringLength())
    return false;
  out.assign(v.GetString(), v.GetStringLength());
  return out.find('\0') == std::string::npos;
}

bool Keys(const Json &v, std::initializer_list<const char *> allowed) {
  if (!v.IsObject())
    return false;
  std::unordered_set<std::string> seen;
  for (auto i = v.MemberBegin(); i != v.MemberEnd(); ++i) {
    std::string key(i->name.GetString(), i->name.GetStringLength());
    if (!seen.insert(key).second ||
        std::none_of(allowed.begin(), allowed.end(), [&](const char *s) { return key == s; }))
      return false;
  }
  return true;
}

bool MemberNumber(const Json &v, const char *key, double &out) {
  return v.HasMember(key) && Number(v[key], out);
}
}  // namespace

Vector Transform::Rotate(Vector v) const {
  return {rotation[0] * v.x + rotation[1] * v.y + rotation[2] * v.z,
          rotation[3] * v.x + rotation[4] * v.y + rotation[5] * v.z,
          rotation[6] * v.x + rotation[7] * v.y + rotation[8] * v.z};
}

Vector Transform::Point(Vector v) const {
  return Add(Rotate(v), translation);
}

Vector Transform::InverseRotate(Vector v) const {
  return {rotation[0] * v.x + rotation[3] * v.y + rotation[6] * v.z,
          rotation[1] * v.x + rotation[4] * v.y + rotation[7] * v.z,
          rotation[2] * v.x + rotation[5] * v.y + rotation[8] * v.z};
}

Loads BodyLoads(const Loads &world, const Transform &body_to_world, Vector reference_body) {
  Vector force = body_to_world.InverseRotate(world.force);
  Vector moment =
      body_to_world.InverseRotate(Add(world.moment, Scale(Cross(body_to_world.translation, world.force), -1)));
  return {force, Add(moment, Scale(Cross(reference_body, force), -1))};
}

std::unique_ptr<Model> Model::Load(const std::string &path, std::string &error) {
  error.clear();
  auto fail = [&](const std::string &message) -> std::unique_ptr<Model> {
    error = message;
    return {};
  };
  std::ifstream file(std::filesystem::u8path(path));
  if (!file)
    return fail("Cannot open model: " + path);
  rapidjson::IStreamWrapper stream(file);
  rapidjson::Document root;
  root.ParseStream<rapidjson::kParseValidateEncodingFlag>(stream);
  if (root.HasParseError())
    return fail(std::string("Invalid model JSON: ") + rapidjson::GetParseError_En(root.GetParseError()));
  if (!Keys(root, {"version", "units", "reference_point", "controls", "parts"}) || !root.HasMember("version") ||
      !root["version"].IsInt() || root["version"].GetInt() != 1 || !root.HasMember("units") ||
      !root["units"].IsString() || std::string(root["units"].GetString(), root["units"].GetStringLength()) != "metres")
    return fail("Model requires version 1, units metres, and known unique fields");
  auto model = std::make_unique<Model>();
  if (!root.HasMember("reference_point") || !VectorValue(root["reference_point"], model->reference_))
    return fail("reference_point must contain three finite body-local coordinates");
  if (!root.HasMember("controls") || !root["controls"].IsArray() || !root.HasMember("parts") ||
      !root["parts"].IsArray() || root["parts"].Empty())
    return fail("Model requires controls and nonempty parts arrays");
  std::unordered_map<std::string, size_t> control_indices, part_indices;
  for (const auto &v : root["controls"].GetArray()) {
    ControlDimension c;
    if (!Keys(v, {"name", "min", "max", "default"}) || !v.HasMember("name") || !Text(v["name"], c.name) ||
        !MemberNumber(v, "min", c.minimum) || !MemberNumber(v, "max", c.maximum) ||
        !MemberNumber(v, "default", c.initial) || !(c.minimum < c.maximum) || c.initial < c.minimum ||
        c.initial > c.maximum || !control_indices.emplace(c.name, model->controls_.size()).second)
      return fail("Invalid or duplicate control dimension");
    model->controls_.push_back(c);
  }
  std::vector<std::string> parents;
  for (const auto &v : root["parts"].GetArray()) {
    Part p;
    std::string parent;
    if (!Keys(v, {"name", "mesh", "parent", "translation", "rotation_xyzw", "hinge"}) || !v.HasMember("name") ||
        !Text(v["name"], p.name) || !v.HasMember("mesh") || !Text(v["mesh"], p.mesh) ||
        !part_indices.emplace(p.name, model->parts_.size()).second)
      return fail("Invalid or duplicate mesh part");
    p.mesh =
        (std::filesystem::u8path(path).parent_path() / std::filesystem::u8path(p.mesh)).lexically_normal().u8string();
    if (v.HasMember("parent") && !v["parent"].IsNull() && !Text(v["parent"], parent))
      return fail("Invalid parent for " + p.name);
    if (v.HasMember("translation") && !VectorValue(v["translation"], p.bind.translation))
      return fail("Invalid translation for " + p.name);
    if (v.HasMember("rotation_xyzw")) {
      const auto &q = v["rotation_xyzw"];
      double x, y, z, w;
      if (!q.IsArray() || q.Size() != 4 || !Number(q[0], x) || !Number(q[1], y) || !Number(q[2], z) ||
          !Number(q[3], w) || std::abs(x * x + y * y + z * z + w * w - 1) > 1e-8)
        return fail("rotation_xyzw must be a unit quaternion for " + p.name);
      p.bind.rotation = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                         2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                         2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
    }
    if (v.HasMember("hinge")) {
      const auto &h = v["hinge"];
      if (!Keys(h, {"pivot", "axis", "offset_deg", "min_deg", "max_deg", "weights"}) || !h.HasMember("pivot") ||
          !VectorValue(h["pivot"], p.pivot) || !h.HasMember("axis") || !VectorValue(h["axis"], p.axis) ||
          !MemberNumber(h, "offset_deg", p.offset) || !MemberNumber(h, "min_deg", p.minimum) ||
          !MemberNumber(h, "max_deg", p.maximum) || p.minimum > p.maximum || p.offset < p.minimum ||
          p.offset > p.maximum || !h.HasMember("weights") || !h["weights"].IsObject())
        return fail("Invalid hinge for " + p.name);
      double length = std::hypot(p.axis.x, p.axis.y, p.axis.z);
      if (!(length > 1e-12) || !std::isfinite(length))
        return fail("Hinge axis must be nonzero for " + p.name);
      p.axis = Scale(p.axis, 1 / length);
      std::unordered_set<std::string> seen;
      for (auto i = h["weights"].MemberBegin(); i != h["weights"].MemberEnd(); ++i) {
        std::string name(i->name.GetString(), i->name.GetStringLength());
        auto control = control_indices.find(name);
        double weight;
        if (control == control_indices.end() || !seen.insert(name).second || !Number(i->value, weight))
          return fail("Unknown, duplicate or invalid hinge control: " + name);
        p.weights.push_back({control->second, weight});
      }
      p.hinged = true;
    }
    model->parts_.push_back(std::move(p));
    parents.push_back(std::move(parent));
  }
  for (size_t i = 0; i < parents.size(); ++i)
    if (!parents[i].empty()) {
      auto p = part_indices.find(parents[i]);
      if (p == part_indices.end())
        return fail("Unknown parent: " + parents[i]);
      model->parts_[i].parent = int(p->second);
    }
  std::vector<int> visited(model->parts_.size());
  std::function<bool(size_t)> visit = [&](size_t i) {
    if (visited[i] == 1)
      return false;
    if (visited[i] == 2)
      return true;
    visited[i] = 1;
    int parent = model->parts_[i].parent;
    if (parent >= 0 && !visit(size_t(parent)))
      return false;
    visited[i] = 2;
    model->order_.push_back(i);
    return true;
  };
  for (size_t i = 0; i < visited.size(); ++i)
    if (!visit(i))
      return fail("Cyclic part hierarchy");
  return model;
}

std::vector<double> Model::DefaultState() const {
  std::vector<double> result;
  for (const auto &c : controls_)
    result.push_back(c.initial);
  return result;
}

bool Model::Evaluate(const std::vector<double> &state, std::vector<PartPose> &poses, std::string &error) const {
  error.clear();
  if (state.size() != controls_.size()) {
    error = "Control state dimension mismatch";
    return false;
  }
  for (size_t i = 0; i < state.size(); ++i)
    if (!std::isfinite(state[i]) || state[i] < controls_[i].minimum || state[i] > controls_[i].maximum) {
      error = "Control outside range: " + controls_[i].name;
      return false;
    }
  std::vector<PartPose> result(parts_.size());
  for (size_t i : order_) {
    const auto &p = parts_[i];
    auto &pose = result[i];
    Transform local = p.bind;
    if (p.hinged) {
      double requested = p.offset;
      for (const auto &weight : p.weights)
        requested += weight.degrees_per_unit * state[weight.control];
      if (!std::isfinite(requested)) {
        error = "Nonfinite hinge angle for " + p.name;
        return false;
      }
      pose.angle_degrees = std::clamp(requested, p.minimum, p.maximum);
      pose.limited = pose.angle_degrees != requested;
      local = Compose(p.bind, Hinge(p.axis, p.pivot, pose.angle_degrees));
    }
    pose.local_to_body = p.parent < 0 ? local : Compose(result[p.parent].local_to_body, local);
  }
  poses.swap(result);
  return true;
}
}  // namespace wind3d::aero
