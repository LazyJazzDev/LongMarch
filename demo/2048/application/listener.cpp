#include "listener.h"

Listener::Listener(Application *application) : application_(application) {
  application_->RegisterListener(this);
}

void Listener::OnCursorEnter(int enter) {
}

void Listener::OnCursorPos(double xpos, double ypos) {
}

void Listener::OnMouseButton(int mouse_button, int state, int mods) {
}

void Listener::OnWindowSize(int width, int height) {
}

Listener::~Listener() {
  application_->UnregisterListener(this);
}
