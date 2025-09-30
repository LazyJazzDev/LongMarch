#pragma once
#include "grassland/grassland.h"
#include "sparkium/core/code_lines.h"

namespace sparkium {

using namespace grassland;

class Core;
class Geometry;
class Material;
class Entity;
class Film;
class Scene;
class Camera;

class Object {
 public:
  virtual ~Object() = default;
  std::unique_ptr<Object> next{nullptr};
  template <typename T>
  T *GetComponent() {
    if (auto ptr = dynamic_cast<T *>(this)) {
      return ptr;
    }
    return next ? next->GetComponent<T>() : nullptr;
  }

  template <typename T, typename... Args>
  T *AddComponent(Args &&...args) {
    auto &tail = FindTail();
    auto obj = std::make_unique<T>(std::forward<Args>(args)...);
    auto res = obj.get();
    tail = std::move(obj);
    return res;
  }

 private:
  std::unique_ptr<Object> &FindTail() {
    if (!next)
      return next;
    return next->FindTail();
  }
};

#define COMPONENT_CAST(obj, cast_type)                   \
  if (auto component = obj->GetComponent<cast_type>()) { \
    return component;                                    \
  } else {                                               \
    return obj->AddComponent<cast_type>(*obj);           \
  }

#define DEDICATED_CAST(obj, is_type, cast_type)            \
  if (auto ptr = dynamic_cast<is_type *>(obj)) {           \
    if (auto component = ptr->GetComponent<cast_type>()) { \
      return component;                                    \
    } else {                                               \
      return ptr->AddComponent<cast_type>(*ptr);           \
    }                                                      \
  }

typedef enum RenderPipeline {
  RENDER_PIPELINE_RASTERIZATION = 0,
  RENDER_PIPELINE_RAY_TRACING = 1,
  RENDER_PIPELINE_AUTO = 2  // Uae raytracing if it could
} RenderPipeline;

}  // namespace sparkium
