#pragma once

#include "listener.h"

class Button : public Listener {
 public:
  Button(Application *application, float left, float top, float right, float bottom);
  virtual ~Button();
  void OnCursorEnter(int enter) final;
  void OnMouseButton(int mouse_button, int state, int mods) final;
  void OnCursorPos(double xpos, double ypos) final;
  void OnWindowSize(int width, int height) final;
  void Resize(float left, float top, float right, float bottom);
  virtual void OnStateChange(int state);
  virtual void OnClick();
  virtual void OnResize();

  void SetClipBounds(glm::vec4 bounds);

  void Activate();
  void Deactivate();

 protected:
  void SetState(int state);

  void CalculateListenerBounds();
  bool IsInsideListenerBounds(float x, float y) const;

  glm::mat3 local_to_world_{};
  Application *app_{nullptr};
  int state_{0};
  float left_{0};
  float top_{0};
  float right_{0};
  float bottom_{0};

  glm::vec4 clip_bounds_{-1.0e9f, -1.0e9f, 1.0e9f, 1.0e9f};

  float listener_left_{0};
  float listener_top_{0};
  float listener_right_{0};
  float listener_bottom_{0};
};
