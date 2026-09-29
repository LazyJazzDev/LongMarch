#include "sparkium/camera/camera_thin_lens.h"

namespace sparkium {

CameraThinLens::CameraThinLens(Core *core, const glm::mat4 &view, float fovy, float aspect)
    : Camera(core, view, fovy, aspect) {
}

}  // namespace sparkium
