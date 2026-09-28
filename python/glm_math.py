"""GLM-compatible matrix helpers for the Python demos (right-handed, like the C++ demos).

Matrices are row-major numpy arrays acting on column vectors; as_bytes() uploads them in
GLM's column-major layout, so shaders shared with the C++ demos read the same values.
"""

import math

import numpy as np


def rotate(angle, axis):
    """glm::rotate(glm::mat4{1.0f}, angle, axis)."""
    x, y, z = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return np.array([
        [c + x * x * t, x * y * t - z * s, x * z * t + y * s, 0],
        [y * x * t + z * s, c + y * y * t, y * z * t - x * s, 0],
        [z * x * t - y * s, z * y * t + x * s, c + z * z * t, 0],
        [0, 0, 0, 1],
    ], np.float32)


def rotate_y(angle):
    return rotate(angle, (0.0, 1.0, 0.0))


def translate(x, y, z):
    m = np.identity(4, np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag(np.array([x, y, z, 1], np.float32))


def look_at(eye, center, up):
    eye, center, up = (np.asarray(v, np.float32) for v in (eye, center, up))
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = (-s @ eye, -u @ eye, f @ eye)
    return m


def perspective(fovy, aspect, near, far, zero_to_one=False):
    """glm::perspective (depth -1..1), or glm::perspectiveZO with zero_to_one."""
    t = math.tan(fovy / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = 1 / (aspect * t)
    m[1, 1] = 1 / t
    m[3, 2] = -1
    if zero_to_one:
        m[2, 2] = far / (near - far)
        m[2, 3] = -(far * near) / (far - near)
    else:
        m[2, 2] = -(far + near) / (far - near)
        m[2, 3] = -(2 * far * near) / (far - near)
    return m


def as_bytes(*matrices):
    return b"".join(np.ascontiguousarray(m.T, np.float32).tobytes() for m in matrices)
