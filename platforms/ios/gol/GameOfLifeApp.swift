import SwiftUI

// The standalone Game of Life app: the same game page as LongMarch Demos,
// without the demo browser or its status header.
@main
struct GameOfLifeApp: App {
  @UIApplicationDelegateAdaptor(GameAppDelegate.self) private var appDelegate
  var body: some Scene {
    WindowGroup {
      GamesView(life: true, onExit: nil)
        .preferredColorScheme(.dark)
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
    }
  }
}
