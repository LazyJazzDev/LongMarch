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

TEST(SimulationClock, LightningRunsExactlyOneGenerationPerFrame) {
  SimulationClock clock;
  int generations = 0;
  for (double elapsed : {0.0, 0.001, 0.016, 1.0, 30.0}) {
    const int previous = generations;
    clock.Advance(elapsed, true, SimulationClock::kLightning, [&] { ++generations; });
    EXPECT_EQ(generations, previous + 1);
  }
  // Switching back cannot inherit any accumulated lightning time.
  clock.Advance(0.0, true, 0, [&] { FAIL() << "Unexpected timed generation"; });
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
