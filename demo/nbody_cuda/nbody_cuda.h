#pragma once
#include "glm/glm.hpp"
#include "params.h"

void UpdateStep(glm::vec3 *positions, glm::vec3 *velocities, int n_particles, float delta_t);
