#include "sparks/core/material.h"

#include "core.h"

namespace sparks {

Material::Material(Core *core, const MaterialLambertian &material) : core_(core), material(material) {
  core_->GraphicsCore()->CreateShader(core_->GetShadersVFS(), "material.hlsl", "CallableMain", "lib_6_3",
                                      &callable_shader_);
  core_->GraphicsCore()->CreateBuffer(sizeof(this->material), graphics::BUFFER_TYPE_STATIC, &material_buffer_);
  material_buffer_->UploadData(&this->material, sizeof(this->material));
}

}  // namespace sparks
