#pragma once
#include "sparkium/backend/graphics/raster/material/material_lambertian.h"
#include "sparkium/backend/graphics/raster/material/material_light.h"
#include "sparkium/backend/graphics/raster/material/material_principled.h"
#include "sparkium/backend/graphics/raster/material/material_specular.h"

namespace sparkium::raster {

Material *DedicatedCast(sparkium::Material *material);

}
