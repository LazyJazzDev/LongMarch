#pragma once

#include <functional>

struct GLFWwindow;

void *InstallTrackpadGestures(GLFWwindow *window, std::function<bool(float)> on_magnify);
void RemoveTrackpadGestures(void *monitor);
