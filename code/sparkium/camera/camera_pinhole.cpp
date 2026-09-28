#include "sparkium/camera/camera_pinhole.h"

namespace sparkium {

CameraPinhole::CameraPinhole(Core *core, const glm::mat4 &view, float fovy, float aspect)
    : Camera(core, view, fovy, aspect) {
}

}  // namespace sparkium
