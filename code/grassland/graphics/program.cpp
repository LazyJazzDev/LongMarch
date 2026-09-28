#include "grassland/graphics/program.h"

namespace grassland::graphics {

void RayTracingProgram::AddHitGroup(Shader *closest_hit_shader,
                                    Shader *any_hit_shader,
                                    Shader *intersection_shader,
                                    bool procedure) {
  HitGroup hit_group{
      closest_hit_shader,
      any_hit_shader,
      intersection_shader,
      procedure,
  };

  AddHitGroup(hit_group);
}

}  // namespace grassland::graphics
