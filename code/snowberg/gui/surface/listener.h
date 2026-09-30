#pragma once
#include "application.h"

namespace snowberg::gui::surface {

class Listener {
 public:
  explicit Listener(Application *application);
  virtual ~Listener();
  virtual void OnCursorEnter(int enter);
  virtual void OnFocus(bool focused);
  virtual void OnCursorPos(double xpos, double ypos);
  virtual void OnMouseButton(int mouse_button, int state, int mods);
  virtual void OnWindowSize(int width, int height);

 protected:
  Application *application_;
};

}  // namespace snowberg::gui::surface
