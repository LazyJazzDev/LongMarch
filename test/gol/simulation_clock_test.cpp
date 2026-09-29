#include "../../demo/gol/simulation_clock.h"

#include <gtest/gtest.h>

TEST(SimulationClock, NormalSpeedsKeepTheirGenerationRates) {
  for (int speed = 0; speed < 3; ++speed) {
    SimulationClock clock;
    int generations = 0;
    clock.Advance(1.0, true, speed, [&] { ++generations; });
    EXPECT_EQ(generations, speed == 0 ? 2 : speed == 1 ? 4 : 10);
  }
}

namespace {
// Generations produced by `frames` frames of the given interval in lightning mode.
int LightningGenerations(double interval, int frames, double jitter = 0.0) {
  SimulationClock clock;
  int generations = 0;
  for (int i = 0; i < frames; ++i) {
    const double elapsed = interval + (i % 2 ? jitter : -jitter);
    const int previous = generations;
    clock.Advance(i ? elapsed : 0.0, true, SimulationClock::kLightning, [&] { ++generations; });
    EXPECT_LE(generations - previous, 1) << "More than one generation in a frame";
  }
  return generations;
}
}  // namespace

TEST(SimulationClock, LightningStartsImmediately) {
  SimulationClock clock;
  int generations = 0;
  clock.Advance(0.0, true, SimulationClock::kLightning, [&] { ++generations; });
  EXPECT_EQ(generations, 1);
  // Switching back cannot inherit any accumulated lightning time.
  clock.Advance(0.0, true, 0, [&] { FAIL() << "Unexpected timed generation"; });
}

TEST(SimulationClock, LightningStepsEveryFrameAtSixtyHertzDespiteJitter) {
  EXPECT_EQ(LightningGenerations(1.0 / 60.0, 600), 600);
  EXPECT_EQ(LightningGenerations(1.0 / 60.0, 600, 0.002), 600);
  // A panel slightly slower than 60 Hz still advances on every frame.
  EXPECT_EQ(LightningGenerations(1.0 / 59.9, 600), 600);
}

TEST(SimulationClock, LightningKeepsSixtyGenerationsPerSecondOnFasterDisplays) {
  for (double hertz : {90.0, 120.0, 144.0}) {
    const int frames = int(hertz * 10);
    EXPECT_NEAR(LightningGenerations(1.0 / hertz, frames), 600, 2) << hertz << " Hz";
  }
}

TEST(SimulationClock, LightningDoesNotCatchUpAfterSlowFrames) {
  // Slow frames advance once each instead of jumping several generations.
  EXPECT_EQ(LightningGenerations(1.0 / 30.0, 300), 300);
  EXPECT_EQ(LightningGenerations(1.0, 10), 10);
}

TEST(SimulationClock, PauseStopsLightningAndDropsTimedBacklog) {
  SimulationClock clock;
  int generations = 0;
  auto step = [&] { ++generations; };
  clock.Advance(0.4, true, 0, step);
  clock.Advance(5.0, false, SimulationClock::kLightning, step);
  clock.Advance(0.2, true, 0, step);
  EXPECT_EQ(generations, 0);
  clock.Advance(0.3, true, 0, step);
  EXPECT_EQ(generations, 1);
}
