#include <iostream>
#include <vector>

#include "snowberg/tui/tui.h"

class Probe : public snowberg::tui::Core {
 public:
  std::vector<std::string> events;
  bool valid{true};

 protected:
  void OnStart() override {
    valid &= Run() == snowberg::tui::RunResult::Error;
    SetForegroundColor(1, 2, 3);
    SetBackgroundColor(4, 5, 6);
    valid &= UpdateRow(0, Columns() - 4, u8"A中文B") == 4;
    valid &= UpdateRow(1, Columns() - 1, u8"中") == 0;
    valid &= UpdateRow(-1, 0, "invalid") == 0;
    valid &= UpdateRow(0, -1, "invalid") == 0;
    valid &= UpdateRow(Rows(), 0, "invalid") == 0;
    valid &= UpdateRow(0, Columns(), "invalid") == 0;
    valid &= UpdateRow(Rows() - 1, Columns() - 1, "Z") == 1;
    ResetColors();
    UpdateRow(2, 0, "CORE_READY");
    events.push_back("start");
  }

  void OnResize(int columns, int rows) override {
    valid &= Columns() == columns && Rows() == rows;
    UpdateRow(2, 0, "SIZE" + std::to_string(columns) + ":" + std::to_string(rows));
    events.push_back("resize:" + std::to_string(columns) + ":" + std::to_string(rows));
  }

  void OnKey(const snowberg::tui::KeyEvent &event) override {
    events.push_back("key:" + std::to_string(static_cast<unsigned>(event.character)));
    if (event.key == snowberg::tui::Key::Enter)
      Terminate();
  }

  void OnMouse(const snowberg::tui::MouseEvent &event) override {
    events.push_back("mouse:" + std::to_string(static_cast<int>(event.action)) + ":" + std::to_string(event.column) +
                     ":" + std::to_string(event.row));
  }

  void OnStop() override {
    events.push_back("stop");
  }
};

int main() {
  Probe probe;
  const auto result = probe.Run();
  if (result == snowberg::tui::RunResult::Unavailable) {
    std::cout << "UNAVAILABLE\n";
    return probe.events.empty() ? 0 : 1;
  }
  for (const auto &event : probe.events)
    std::cout << event << '\n';
  return result == snowberg::tui::RunResult::Completed && probe.valid && probe.UpdateRow(0, 0, "inactive") == 0 ? 0 : 1;
}
