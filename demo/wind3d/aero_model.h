#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace wind3d::aero {
struct Vector {
  double x{}, y{}, z{};
};

// Row-major rigid transform from local coordinates to the parent frame.
struct Transform {
  std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  Vector translation;
  Vector Rotate(Vector v) const;
  Vector Point(Vector v) const;
  Vector InverseRotate(Vector v) const;
};

struct ControlDimension {
  std::string name;
  double minimum{}, maximum{}, initial{};
};

struct ControlWeight {
  size_t control{};
  double degrees_per_unit{};
};

struct Part {
  std::string name, mesh;
  int parent{-1};
  Transform bind;
  Vector pivot{}, axis{0, 0, 1};
  double offset{}, minimum{}, maximum{};
  std::vector<ControlWeight> weights;
  bool hinged{};
};

struct PartPose {
  Transform local_to_body;
  double angle_degrees{};
  bool limited{};
};

struct Loads {
  Vector force, moment;
};

// Input moment is about the world origin. Output is in body axes about the
// requested body-local reference point; translation and rotation both matter.
Loads BodyLoads(const Loads &world, const Transform &body_to_world, Vector reference_body);

class Model {
 public:
  static std::unique_ptr<Model> Load(const std::string &path, std::string &error);

  const std::vector<ControlDimension> &Controls() const {
    return controls_;
  }

  const std::vector<Part> &Parts() const {
    return parts_;
  }

  Vector ReferencePoint() const {
    return reference_;
  }

  std::vector<double> DefaultState() const;
  bool Evaluate(const std::vector<double> &state, std::vector<PartPose> &poses, std::string &error) const;

 private:
  std::vector<ControlDimension> controls_;
  std::vector<Part> parts_;
  std::vector<size_t> order_;
  Vector reference_;
};
}  // namespace wind3d::aero
