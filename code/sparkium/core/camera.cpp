#include "sparkium/core/camera.h"

#include "core.h"

namespace sparkium {
Camera::Camera(Core *core, const glm::mat4 &view, float fovy, float aspect)
    : core_(core), view(view), fovy(fovy), aspect(aspect) {
}
}  // namespace sparkium
