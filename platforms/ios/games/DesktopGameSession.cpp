#include "DesktopGameSession.h"

#include <cstdlib>

#include "demo/2048/2048.h"
#include "demo/gol/game_of_life_gui.h"

DesktopGameSession::DesktopGameSession(const std::string &name, grassland::graphics::BackendAPI backend) {
  if (name == "gol") {
    const char *size = std::getenv("LONGMARCH_SMOKE_GOL_SIZE");
    const int extent = size ? std::clamp(std::atoi(size), grid_size::kMin, grid_size::kMax) : 0;
    life_ = std::make_unique<life_demo::GameOfLife>("Game of Life", 1280, 720, extent ? extent : grid_size::kDefault,
                                                    extent ? extent : grid_size::kDefault, backend, true);
    if (extent) {
      life_->SetRandomInitialCells(.3f, 42);
      life_->SetInitialPlaying(true);
    }
    life_->InitializeHosted();
  } else {
    puzzle_ = std::make_unique<puzzle_demo::TwentyFourEight>("2048", 720, 960, backend, true);
    puzzle_->InitializeHosted();
  }
}

DesktopGameSession::~DesktopGameSession() {
  if (life_)
    life_->CloseHosted();
  if (puzzle_)
    puzzle_->CloseHosted();
}

void DesktopGameSession::Render() {
  if (life_)
    life_->RenderHostedFrame();
  else
    puzzle_->RenderHostedFrame();
}

void DesktopGameSession::Resize(int width, int height) {
  Window()->UpdateHostedSize({width, height}, {width, height});
}

grassland::graphics::Core *DesktopGameSession::Core() const {
  return life_ ? life_->Core() : puzzle_->Core();
}

grassland::graphics::Image *DesktopGameSession::Image() const {
  return life_ ? life_->PresentedImage() : puzzle_->PresentedImage();
}

grassland::graphics::Window *DesktopGameSession::Window() const {
  return life_ ? life_->GetWindow() : puzzle_->GetWindow();
}

int DesktopGameSession::FileRequest() const {
  return life_ ? life_->HostedFileRequest() : 0;
}

std::string DesktopGameSession::CompleteFile(const std::string &path) {
  return life_ ? life_->CompleteHostedFile(path) : "";
}

void DesktopGameSession::ResetClock() {
  if (life_)
    life_->ResetFrameClock();
  else
    puzzle_->ResetFrameClock();
}

double DesktopGameSession::NextFrameDelay() const {
  return life_ ? life_->NextFrameDelay() : puzzle_->NextFrameDelay();
}

void DesktopGameSession::PrepareInput(bool sleeping) {
  if (sleeping)
    ResetClock();
  else if (life_)
    life_->ResetAnimationClock();
}

void DesktopGameSession::SetIconOrientation(float radians) {
  if (life_)
    life_->SetIconOrientation(radians);
}

void DesktopGameSession::SetBottomControlInset(float height_fraction) {
  if (life_)
    life_->SetBottomControlInset(height_fraction);
}

void DesktopGameSession::SetCutoutInsets(float left, float top, float right) {
  if (life_)
    life_->SetCutoutInsets(left, top, right);
}

void DesktopGameSession::SetControlExtentLimit(float height_fraction) {
  if (life_)
    life_->SetControlExtentLimit(height_fraction);
}

void DesktopGameSession::EnableNativeSizeControls() {
  if (life_)
    life_->EnableNativeSizeControls();
}

glm::ivec2 DesktopGameSession::TakeSizeControlRequest() {
  return life_ ? life_->TakeSizeControlRequest() : glm::ivec2{0};
}

glm::vec4 DesktopGameSession::SizeControlBounds(int axis) const {
  return life_ ? life_->SizeControlBounds(axis) : glm::vec4{0.0f};
}

void DesktopGameSession::SetGridDimension(int axis, int value) {
  if (life_)
    life_->SetGridDimension(axis, value);
}
