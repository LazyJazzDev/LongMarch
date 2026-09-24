import SwiftUI
import UIKit

struct DemoItem: Identifiable {
  let id: String
  let title: String
  let subtitle: String
  let symbol: String
  var unavailable: String? = nil

  static let graphics = [
    DemoItem(
      id: "graphics_hello_cube", title: "Hello Cube", subtitle: "Depth-tested rotating cube",
      symbol: "cube"),
    DemoItem(
      id: "graphics_hello_hdr", title: "Hello HDR",
      subtitle: "Linear EDR gradient and SDR reference white", symbol: "sun.max.fill"),
    DemoItem(
      id: "graphics_hello_ray_query", title: "Hello Ray Query",
      subtitle: "Animated triangle and procedural sphere", symbol: "rays"),
    DemoItem(
      id: "graphics_hello_external_shader", title: "External Shader",
      subtitle: "External RT pipeline shaders", symbol: "doc.text",
      unavailable: "Requires RT shader pipelines, unavailable on Metal; use Ray Query."),
    DemoItem(
      id: "graphics_hello_triangle", title: "Hello Triangle",
      subtitle: "Indexed geometry and vertex colors", symbol: "triangle"),
    DemoItem(
      id: "graphics_hello_texture", title: "Hello Texture",
      subtitle: "Texture sampling and depth attachment", symbol: "square.on.square"),
    DemoItem(
      id: "graphics_hello_blend", title: "Hello Blend",
      subtitle: "Overlapping triangles and alpha blending", symbol: "square.3.layers.3d"),
    DemoItem(
      id: "graphics_hello_resize", title: "Hello Resize",
      subtitle: "Rotating cube and resizable render targets",
      symbol: "arrow.up.left.and.arrow.down.right"),
    DemoItem(
      id: "graphics_hello_sdr_sample", title: "Hello SDR Sample",
      subtitle: "Pixel patterns and display response", symbol: "sun.max"),
    DemoItem(
      id: "graphics_hello_raytracing", title: "Hello Ray Tracing",
      subtitle: "Native ray tracing pipeline", symbol: "rays",
      unavailable:
        "Requires a full RT Pipeline; the Metal backend currently supports Ray Query. Use Sparkium to try hardware ray tracing."
    ),
    DemoItem(
      id: "graphics_rt_multi_shader_group", title: "RT Multi Shader Group",
      subtitle: "Procedural intersections and callable shaders", symbol: "cube.transparent",
      unavailable:
        "Requires RT shader groups and procedural intersection shaders, which the current Metal backend does not implement."
    ),
  ]
  static let games = [
    DemoItem(
      id: "2048", title: "2048", subtitle: "Swipe to merge · Expectimax AI",
      symbol: "square.grid.2x2"),
    DemoItem(
      id: "gol", title: "Game of Life", subtitle: "Editable cells · Boundary modes · Pattern files",
      symbol: "squareshape.split.3x3"),
  ]
  static let compute = [
    DemoItem(
      id: "nbody_cs", title: "NBody CS",
      subtitle: "GPU gravity simulation with interactive galaxies", symbol: "sparkles")
  ]
  static let rendering = [
    DemoItem(
      id: "sparkium", title: "Sparkium", subtitle: "Path tracing · 9 bundled scenes", symbol: "cube"
    )
  ]
  static let all = games + graphics + compute + rendering
}

private struct DemoBrowser: View {
  @State private var selected: DemoItem? = {
    let env = ProcessInfo.processInfo.environment
    let id = env["LONGMARCH_SMOKE_DEMO"] ?? (env["SPARKIUM_SMOKE_SCENE"] == nil ? "" : "sparkium")
    return DemoItem.all.first { $0.id == id && $0.unavailable == nil }
  }()
  var body: some View {
    Group {
      if let selected {
        if selected.id == "sparkium" {
          SparkiumDemoView { self.selected = nil }
        } else if selected.id == "2048" || selected.id == "gol" {
          GamesView(life: selected.id == "gol") { self.selected = nil }.id(selected.id)
        } else {
          GraphicsDemoView(demo: selected) { self.selected = nil }.id(selected.id)
        }
      } else {
        NavigationStack {
          List {
            Section("Games") { rows(DemoItem.games) }
            Section("Rendering") { rows(DemoItem.rendering) }
            Section("Compute") { rows(DemoItem.compute) }
            Section("Graphics") { rows(DemoItem.graphics.filter { $0.unavailable == nil }) }
            Section("Unavailable on Metal") {
              rows(DemoItem.graphics.filter { $0.unavailable != nil })
            }
          }
          .navigationTitle("LongMarch Demos")
          .navigationBarTitleDisplayMode(.inline)
          .accessibilityIdentifier("demo-list")
        }
      }
    }
    .preferredColorScheme(.dark)
    .statusBarHidden()
    .persistentSystemOverlays(.hidden)
  }
  private func rows(_ demos: [DemoItem]) -> some View {
    ForEach(demos) { demo in
      Button {
        selected = demo
      } label: {
        HStack(spacing: 16) {
          Image(systemName: demo.symbol).font(.title2).frame(width: 32).foregroundStyle(.tint)
          VStack(alignment: .leading, spacing: 4) {
            Text(demo.title).font(.headline)
            Text(demo.unavailable ?? demo.subtitle).font(.subheadline).foregroundStyle(.secondary)
          }
          Spacer()
          Image(systemName: demo.unavailable == nil ? "chevron.right" : "lock").foregroundStyle(
            .secondary)
        }.padding(.vertical, 5)
      }
      .buttonStyle(.plain)
      .disabled(demo.unavailable != nil)
      .accessibilityIdentifier("demo-" + demo.id)
    }
  }
}

// GoL freezes the interface orientation for the lifetime of its page. Device
// orientation remains available to its icons without rotating the UIWindow.
@MainActor
final class DemoAppDelegate: NSObject, UIApplicationDelegate {
  static var orientationMask: UIInterfaceOrientationMask = .allButUpsideDown

  func application(
    _ application: UIApplication, supportedInterfaceOrientationsFor window: UIWindow?
  ) -> UIInterfaceOrientationMask {
    Self.orientationMask
  }
}

@main
struct LongMarchDemosApp: App {
  @UIApplicationDelegateAdaptor(DemoAppDelegate.self) private var appDelegate
  var body: some Scene { WindowGroup { DemoBrowser() } }
}
