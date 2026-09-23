#include <map>
#include <memory>

#include "2048_lib.h"
#include "application/text_bar.h"
#include "block_renderer.h"

struct NumberBlock {
  enum class Stage { kStop, kAppear, kMove, kMerge, kBeMerged, kDead };
  int x;
  int y;
  Stage stage{Stage::kAppear};
  float pos_x{0.0f};
  float pos_y{0.0f};
  int number{0};
  NumberBlock(int x, int y, int number);
  [[nodiscard]] int GetNumber() const;

  void Merge(int new_x, int new_y, int new_number);
  void BeMerged(int x, int y);
  void Move(int x, int y);
  void Kill();
  [[nodiscard]] bool IsDead() const;
  void FinishTurn();
  void Render(BlockRenderer *block_renderer, float alpha, float depth_offset) const;
  [[nodiscard]] Block GetBlock() const;
};
