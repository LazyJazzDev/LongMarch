#include "button.h"

Button::Button(Application *app, float left, float top, float right, float bottom)
    : Listener(app),
      app_(app),
      left_(left),
      top_(top),
      right_(right),
      bottom_(bottom) {
  if (top_ > bottom_)
    std::swap(top_, bottom_);

  CalculateListenerBounds();
}

Button::~Button() {
  app_->UnregisterListener(this);
}

void Button::OnCursorEnter(int enter) {
  if (!enter) {
    if (state_) {
      SetState(0);
    }
  }
}

void Button::OnMouseButton(int mouse_button, int state, int mods) {
  const auto local_pos = app_->GetWindow()->GetCursorPosition();
  bool inside = IsInsideListenerBounds(local_pos.x, local_pos.y);
  if (mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
    if (state == GLFW_RELEASE) {
      if (inside) {
        if (state_ == 2) {
          OnClick();
        }
        SetState(1);
      }
    } else if (state == GLFW_PRESS) {
      if (inside) {
        SetState(2);
      }
    }
  }
}

void Button::OnCursorPos(double xpos, double ypos) {
  auto local_pos = glm::vec2{float(xpos), float(ypos)};
  bool inside = IsInsideListenerBounds(local_pos.x, local_pos.y);
  if (inside) {
    if (!state_) {
      SetState(1);
    }
  } else {
    if (state_) {
      SetState(0);
    }
  }
}

void Button::Resize(float left, float top, float right, float bottom) {
  if (top > bottom)
    std::swap(top, bottom);

  left_ = left;
  top_ = top;
  right_ = right;
  bottom_ = bottom;

  CalculateListenerBounds();

  OnResize();
}

void Button::SetState(int state) {
  state_ = state;
  OnStateChange(state);
}

void Button::OnClick() {
}

void Button::OnResize() {
}

void Button::OnStateChange(int state) {
}

void Button::OnWindowSize(int width, int height) {
  Listener::OnWindowSize(width, height);
}

void Button::Activate() {
  app_->RegisterListener(this);
}

void Button::Deactivate() {
  app_->UnregisterListener(this);
}

void Button::CalculateListenerBounds() {
  // Get frame size
  const auto framebuffer = app_->GetWindow()->GetFramebufferSize();

  // Get window size
  const auto window = app_->GetWindow()->GetSize();

  // Scale from frame to window
  float scale_x = float(window.x) / float(std::max(framebuffer.x, 1));
  float scale_y = float(window.y) / float(std::max(framebuffer.y, 1));

  // Calculate listener bounds
  listener_left_ = left_ * scale_x;
  listener_top_ = top_ * scale_y;
  listener_right_ = right_ * scale_x;
  listener_bottom_ = bottom_ * scale_y;
}

bool Button::IsInsideListenerBounds(float x, float y) const {
  return (listener_left_ <= x) && (x <= listener_right_) && (listener_top_ <= y) && (y <= listener_bottom_);
}
