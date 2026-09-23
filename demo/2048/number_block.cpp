#include "number_block.h"

#include "rounded_rectangle.h"

NumberBlock::NumberBlock(int new_x, int new_y, int new_number) {
  x = new_x;
  y = new_y;
  pos_x = float(x);
  pos_y = float(y);
  stage = Stage::kAppear;
  number = new_number;
}

int NumberBlock::GetNumber() const {
  return number;
}

void NumberBlock::FinishTurn() {
  pos_x = float(x);
  pos_y = float(y);
  if (stage == Stage::kBeMerged) {
    stage = Stage::kDead;
  } else {
    stage = Stage::kStop;
  }
}

bool NumberBlock::IsDead() const {
  return stage == Stage::kDead;
}

void NumberBlock::Move(int new_x, int new_y) {
  x = new_x;
  y = new_y;
  stage = Stage::kMove;
}

void NumberBlock::BeMerged(int new_x, int new_y) {
  x = new_x;
  y = new_y;
  stage = Stage::kBeMerged;
}

void NumberBlock::Kill() {
  stage = Stage::kDead;
}

void NumberBlock::Merge(int new_x, int new_y, int new_number) {
  x = new_x;
  y = new_y;
  number = new_number;
  stage = Stage::kMerge;
}

void NumberBlock::Render(BlockRenderer *block_renderer, float alpha, float depth_offset) const {
  const float appear_time = 0.6f;
  const float move_time = 0.5f;
  float mixed_x = pos_x, mixed_y = pos_y;
  float size = 1.0f;

  float move_alpha = 1.0f;
  if (alpha < move_time) {
    move_alpha = alpha / move_time;
    move_alpha = std::sqrt(move_alpha);
  }
  mixed_x = Mix(pos_x, float(x), move_alpha);
  mixed_y = Mix(pos_y, float(y), move_alpha);

  if (stage == Stage::kAppear) {
    mixed_x = float(x);
    mixed_y = float(y);
    if (alpha < appear_time)
      return;
    size = (alpha - appear_time) / (1.0f - appear_time) * 0.5f + 0.5f;
  } else if (stage == Stage::kMerge) {
    size = 1.0f + alpha * 0.12f;
  }

  block_renderer->Render(number, mixed_x, mixed_y, size, stage != Stage::kMerge ? 1.0f : alpha, depth_offset);
}

Block NumberBlock::GetBlock() const {
  return Block{x, y, number};
}
