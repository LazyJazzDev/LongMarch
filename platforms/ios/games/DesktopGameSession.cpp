#include "DesktopGameSession.h"

#include "demo/2048/2048.h"
#include "demo/gol/game_of_life_gui.h"

DesktopGameSession::DesktopGameSession(const std::string &name) {
  if (name == "gol") {
    life_ = std::make_unique<life_demo::GameOfLife>("Game of Life", 1280, 720, 40, 30,
                                                    grassland::graphics::BACKEND_API_METAL, true);
    life_->InitializeHosted();
  } else {
    puzzle_ =
        std::make_unique<puzzle_demo::TwentyFourEight>("2048", 720, 960, grassland::graphics::BACKEND_API_METAL, true);
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
