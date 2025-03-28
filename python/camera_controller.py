import time
import glfw
glfw.init()
from long_march.grassland import graphics
from long_march.snow_mount import visualizer
from long_march.grassland.math import rotation
import numpy as np
import math
class CameraController:
    def __init__(self, window: graphics.Window, camera: visualizer.Camera):
        self.window = window
        self.camera = camera
        self.camera_position = np.asarray([0., 0.5, 1.])
        self.camera_orientation = np.asarray([-0.6, 0., 0.])
        self.last_update_time = time.time()
        self.last_cursor_pos = self.window.get_cursor_pos()
        self.move_speed = 1.0
        self.rot_speed = 0.003

    def set_camera_position(self, position, orientation):
        self.camera_position = np.asarray(position)
        self.camera_orientation = np.asarray(orientation)

    def update(self, effective=True):
        this_update_time = time.time()
        period = this_update_time - self.last_update_time
        this_cursor_pos = self.window.get_cursor_pos()
        if effective:
            cursor_diff = [this_cursor_pos[0] - self.last_cursor_pos[0], this_cursor_pos[1] - self.last_cursor_pos[1]]
            move_signal = [0., 0., 0.]
            if self.window.get_key(glfw.KEY_A):
                move_signal[0] += -1
            if self.window.get_key(glfw.KEY_D):
                move_signal[0] += 1
            if self.window.get_key(glfw.KEY_W):
                move_signal[2] += -1
            if self.window.get_key(glfw.KEY_S):
                move_signal[2] += 1
            if self.window.get_key(glfw.KEY_SPACE):
                move_signal[1] += 1
            if self.window.get_key(glfw.KEY_LEFT_CONTROL):
                move_signal[1] -= 1
            move_signal = np.asarray(move_signal)
            move_signal = move_signal * self.move_speed * period

            if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
                self.camera_orientation[1] -= cursor_diff[0] * self.rot_speed
                self.camera_orientation[0] -= cursor_diff[1] * self.rot_speed
                self.camera_orientation[0] = np.clip(self.camera_orientation[0], -math.pi / 2, math.pi / 2)
                self.camera_orientation[1] = self.camera_orientation[1] % (2 * math.pi)
            R = np.identity(4)

            R[:3, :3] = rotation([0, self.camera_orientation[1], 0]) @ rotation(
                [self.camera_orientation[0], 0, 0]) @ rotation([0, 0, self.camera_orientation[2]])

            self.camera_position += R[:3, :3] @ move_signal

            R[:3, 3] = self.camera_position

            # inverse matrix R
            R = np.linalg.inv(R)

            self.camera.view = R
        self.last_update_time = this_update_time
        self.last_cursor_pos = this_cursor_pos
