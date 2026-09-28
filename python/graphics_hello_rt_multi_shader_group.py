import sys

import numpy as np
from long_march import graphics

import graphics_hello as gh
from graphics_hello_raytracing import create_triangle_buffers, require_ray_tracing_pipelines

UNIT_AABB = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], np.float32)  # RayTracingAABB


def create_scene_blas(core, aabb_buffer_type=graphics.BUFFER_TYPE_DYNAMIC):
    """Triangle and unit-AABB BLAS objects shared by the multi-shader-group, external-shader and ray-query modules."""
    vertex_buffer, index_buffer = create_triangle_buffers(core)
    aabb_buffer = core.create_buffer(UNIT_AABB.nbytes, aabb_buffer_type)
    aabb_buffer.upload_data(UNIT_AABB.tobytes())
    triangle_blas = core.create_blas(vertex_buffer, index_buffer, 12)
    sphere_blas = core.create_blas(graphics.BufferRange(aabb_buffer), UNIT_AABB.nbytes, 1,
                                   graphics.RAYTRACING_GEOMETRY_FLAG_OPAQUE)
    return triangle_blas, sphere_blas


def scene_instances(triangle_blas, sphere_blas, theta=None):
    """Triangle at x = -2 and a nonuniformly scaled sphere at x = +2, both rotating about y."""
    if theta is None:
        triangle = sphere = np.identity(4, np.float32)
    else:
        triangle = gh.translate(-2.0, 0.0, 0.0) @ gh.rotate_y(theta)
        sphere = gh.translate(2.0, 0.0, 0.0) @ gh.rotate_y(theta) @ gh.scale(1.0, 1.0, 0.5)
    return [triangle_blas.make_instance(gh.instance_transform(triangle), 0, 0xFF, 0),
            sphere_blas.make_instance(gh.instance_transform(sphere), 0, 0xFF, 1)]


class Module(gh.Module):
    title = " Graphics Ray Tracing Multi Shader Group"

    def __init__(self, api=graphics.BACKEND_API_DEFAULT):
        super().__init__()
        self.core = require_ray_tracing_pipelines(api, "Ray tracing pipelines are unavailable on this device")

    def create_shaders(self):
        shader = gh.load_shader("modules/rt_multi_shader_group/shaders/shader.slang")
        self.raygen_shader = self.core.create_shader(shader, "RayGenMain", "lib_6_3")
        self.miss_shader = self.core.create_shader(shader, "MissMain", "lib_6_3")
        self.closest_hit_shader = self.core.create_shader(shader, "ClosestHitMain", "lib_6_3")
        self.sphere_closest_hit_shader = self.core.create_shader(shader, "SphereClosestHitMain", "lib_6_3")
        self.sphere_intersection_shader = self.core.create_shader(shader, "SphereIntersectionMain", "lib_6_3")
        self.callable_shader = self.core.create_shader(shader, "CallableMain", "lib_6_3")

    def on_init(self):
        self.alive = True
        self.window = self.core.create_window(1280, 720, gh.graphics_hello_title(self.core.api()) + self.title)

        self.camera_object_buffer = self.core.create_buffer(2 * 64, graphics.BUFFER_TYPE_DYNAMIC)
        self.camera_object_buffer.upload_data(gh.camera_object(self.window, (0.0, 0.0, 5.0)))

        self.color_image = self.core.create_image(self.window.get_width(), self.window.get_height(),
                                                  graphics.IMAGE_FORMAT_R32G32B32A32_SFLOAT)

        self.create_shaders()
        gh.log_info("Shader compiled successfully")

        self.triangle_blas, self.sphere_blas = create_scene_blas(self.core)
        self.tlas = self.core.create_tlas(scene_instances(self.triangle_blas, self.sphere_blas))

        self.program = self.core.create_raytracing_program()
        self.program.add_ray_gen_shader(self.raygen_shader)
        self.program.add_miss_shader(self.miss_shader)
        self.program.add_hit_group(self.closest_hit_shader)
        self.program.add_hit_group(self.sphere_closest_hit_shader, None, self.sphere_intersection_shader, True)
        self.program.add_callable_shader(self.callable_shader)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_ACCELERATION_STRUCTURE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_WRITABLE_IMAGE, 1)
        self.program.add_resource_binding(graphics.RESOURCE_TYPE_UNIFORM_BUFFER, 1)
        self.program.finalize([0], [0, 1], [0])

    def on_close(self):
        self.core.wait_gpu()
        del self.program, self.raygen_shader, self.miss_shader, self.closest_hit_shader
        del self.sphere_closest_hit_shader, self.sphere_intersection_shader, self.callable_shader
        del self.tlas, self.sphere_blas, self.triangle_blas
        del self.color_image, self.camera_object_buffer

    def on_update(self):
        if self.update_alive():
            self.tlas.update_instances(scene_instances(self.triangle_blas, self.sphere_blas, self.rotation_angle()))

    def on_render(self):
        command_context = self.core.create_command_context()
        command_context.cmd_clear_image(self.color_image, [0.6, 0.7, 0.8, 1.0])
        command_context.cmd_bind_raytracing_program(self.program)
        command_context.cmd_bind_resources(0, self.tlas, graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_bind_resources(1, [self.color_image], graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_bind_resources(2, [self.camera_object_buffer], graphics.BIND_POINT_RAYTRACING)
        command_context.cmd_dispatch_rays(self.window.get_width(), self.window.get_height(), 1)
        command_context.cmd_present(self.window, self.color_image)
        self.core.submit_command_context(command_context)


if __name__ == "__main__":
    sys.exit(gh.main(module="rt_multi_shader_group"))
