#include "../../demo/gol/simulation_clock.h"

#include <gtest/gtest.h>

TEST(SimulationClock, NormalSpeedsKeepTheirGenerationRates) {
  for (int speed = 0; speed < 3; ++speed) {
    SimulationClock clock;
    int generations = 0;
    clock.Advance(1.0, true, speed, [&] { ++generations; }, [] { return 0.0; });
    EXPECT_EQ(generations, speed == 0 ? 2 : speed == 1 ? 4 : 10);
  }
}

TEST(SimulationClock, LightningRunsMultipleGenerationsWithoutElapsedTime) {
  SimulationClock clock;
  int generations = 0;
  double time = 0;
  clock.Advance(
      0.0, true, SimulationClock::kLightning,
      [&] {
        ++generations;
        time += 0.001;
      },
      [&] { return time; });
  EXPECT_GE(generations, 8);
  EXPECT_LE(time, 0.009);
  // Switching back cannot inherit any accumulated lightning time.
  clock.Advance(0.0, true, 0, [&] { FAIL() << "Unexpected timed generation"; }, [] { return 0.0; });
}

TEST(SimulationClock, PauseStopsLightningAndDropsTimedBacklog) {
  SimulationClock clock;
  int generations = 0;
  auto step = [&] { ++generations; };
  auto now = [] { return 0.0; };
  clock.Advance(0.4, true, 0, step, now);
  clock.Advance(5.0, false, SimulationClock::kLightning, step, now);
  clock.Advance(0.2, true, 0, step, now);
  EXPECT_EQ(generations, 0);
  clock.Advance(0.3, true, 0, step, now);
  EXPECT_EQ(generations, 1);
}
