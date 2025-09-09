import math

import numpy as np
from long_march import grassland
from long_march import snowberg
from long_march.grassland import graphics

import open3d as o3d

from camera_controller import CameraController


def main():
    graphics_core_settings = graphics.CoreSettings(frames_in_flight=2)
    graphics_core = graphics.Core(graphics.BACKEND_API_D3D12, graphics_core_settings)
    vis_core = snowberg.visualizer.Core(graphics_core)

    window = graphics_core.create_window(1280, 720, "Visualizer", resizable=True)

    scene = vis_core.create_scene()
    film = vis_core.create_film(1280, 720)
    camera = vis_core.create_camera(proj=snowberg.visualizer.perspective(math.radians(60.0), 1280 / 720, 0.1, 100.0),
                                    view=snowberg.visualizer.look_at([0, 1, 5], [0, 0, 0], [0, 1, 0]))

    def on_resize(width, height):
        nonlocal film, camera
        film = vis_core.create_film(width, height)
        camera.proj = snowberg.visualizer.perspective(math.radians(60.0), width / height, 0.1, 100.0)

    window.reg_resize_callback(on_resize)

    mesh = vis_core.create_mesh()
    # load form "meshes/cube.obj"
    o3d_cube_mesh = o3d.io.read_triangle_mesh(grassland.util.find_asset_file("meshes/cube.obj"))
    mesh.set_vertices(o3d_cube_mesh.vertices)
    mesh.set_indices(np.asarray(o3d_cube_mesh.triangles).flatten())

    entity = vis_core.create_entity_mesh_object(mesh, snowberg.visualizer.Material([0.8, 0.8, 0.8, 1.0]), np.identity(4))
    scene.add_entity(entity)

    ambient_light = vis_core.create_entity_ambient_light([0.5, 0.5, 0.5])
    scene.add_entity(ambient_light)

    light_count = 100
    directional_lights = [vis_core.create_entity_directional_light([3., 1., 2.], np.asarray([0.5, 0.5, 0.5]) / light_count) for _ in range(light_count)]
    for directional_light in directional_lights:
        scene.add_entity(directional_light)

    camera_controller = CameraController(window, camera)
    camera_controller.set_camera_position([0., 1., 5.], [0., 0., 0.])

    fps_counter = grassland.util.FPSCounter()

    while not window.should_close():
        camera_controller.update(True)
        context = graphics_core.create_command_context()
        vis_core.render(context, scene, camera, film)
        context.cmd_present(window, film.get_image())
        context.submit()
        graphics.glfw_poll_events()
        window.set_title("Visualizer - FPS: {:.2f}".format(fps_counter.tick_fps()))


if __name__ == "__main__":
    main()
