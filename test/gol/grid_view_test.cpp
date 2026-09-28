#include "../../demo/gol/grid_view.h"

#include <gtest/gtest.h>

TEST(GridView, ZoomKeepsPointUnderCursor) {
  GridView view;
  view.pan = {20.0f, -30.0f};
  const glm::vec2 anchor{150.0f, 80.0f};
  const auto point = (anchor - view.pan) / view.zoom;
  view.Zoom(2.5f, anchor);
  EXPECT_NEAR((point * view.zoom + view.pan).x, anchor.x, 0.001f);
  EXPECT_NEAR((point * view.zoom + view.pan).y, anchor.y, 0.001f);
  view.Zoom(0.4f, anchor);
  EXPECT_NEAR(view.pan.x, 20.0f, 0.001f);
  EXPECT_NEAR(view.pan.y, -30.0f, 0.001f);
}

TEST(GridView, BoundsKeepSmallGridCenteredAndLargeGridReachable) {
  GridView view;
  view.pan = {10000.0f, -10000.0f};
  view.Clamp({800, 600}, {760, 570});
  EXPECT_EQ(view.pan, glm::vec2(0));
  view.zoom = 2;
  view.pan = {10000.0f, -10000.0f};
  view.Clamp({800, 600}, {760, 570});
  EXPECT_EQ(view.pan, glm::vec2(400, -300));
  view.Clamp({2000, 600}, {760, 570});
  EXPECT_EQ(view.pan, glm::vec2(0, -300));
}

TEST(GridView, EnlargedGridEdgesHaveViewportMarginInBothDirections) {
  GridView view;
  view.zoom = 3;
  const glm::vec2 viewport{800, 600};
  const glm::vec2 fitted{760, 570};
  for (float direction : {-1.0f, 1.0f}) {
    view.pan = glm::vec2{direction * 10000.0f};
    view.Clamp(viewport, fitted);
    const auto edge = viewport * 0.5f + view.pan - direction * fitted * view.zoom * 0.5f;
    const auto margin = direction > 0 ? edge : viewport - edge;
    EXPECT_NEAR(margin.x, viewport.x * 0.05f, 0.001f);
    EXPECT_NEAR(margin.y, viewport.y * 0.05f, 0.001f);
  }
  view.Zoom(1.0f / 3.0f, {100, 100});
  view.Clamp(viewport, fitted);
  EXPECT_EQ(view.pan, glm::vec2(0));
}

TEST(GridView, ZoomLimitsAndReset) {
  GridView view;
  view.Zoom(100, {0, 0});
  EXPECT_EQ(view.zoom, 12);
  view.Zoom(0.001f, {0, 0});
  EXPECT_EQ(view.zoom, 1);
  view.Zoom(3, {100, 100});
  view = {};
  EXPECT_EQ(view.zoom, 1);
  EXPECT_EQ(view.pan, glm::vec2(0));
}
