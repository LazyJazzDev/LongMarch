import ImageIO
import SwiftUI

private struct BundledScene: Decodable, Identifiable {
  let id: String
  let name: String
}

@MainActor
private final class RenderModel: ObservableObject {
  @Published var scenes: [BundledScene] = []
  @Published var selection = "cornell_box"
  @Published var dimension = 256
  @Published var targetSamples = 32
  @Published var image: CGImage?
  @Published var spp = 0
  @Published var elapsed = 0.0
  @Published var backend = "Metal · Path Tracing - Ray Query"
  @Published var status = "Ready"
  @Published var busy = false
  @Published var stopping = false
  @Published var error: String?
  private let renderer = SparkiumRenderer()
  private var resources: URL?
  private var smokeStarted = false
  func startSmokeTestIfRequested() {
    guard !smokeStarted,
      let scene = ProcessInfo.processInfo.environment["SPARKIUM_SMOKE_SCENE"],
      scenes.contains(where: { $0.id == scene })
    else { return }
    smokeStarted = true
    selection = scene
    dimension = 128
    targetSamples = 2
    render()
  }
  private func saveSmokeResult() {
    guard smokeStarted else { return }
    do {
      let directory = try FileManager.default.url(
        for: .documentDirectory, in: .userDomainMask,
        appropriateFor: nil, create: true)
      if let image,
        let destination = CGImageDestinationCreateWithURL(
          directory.appendingPathComponent("SmokeResult.png") as CFURL, "public.png" as CFString, 1,
          nil)
      {
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
      }
      let result: [String: Any] = [
        "scene": selection, "spp": spp, "status": status,
        "error": error ?? "", "backend": backend, "seconds": elapsed,
      ]
      try JSONSerialization.data(withJSONObject: result, options: .prettyPrinted)
        .write(to: directory.appendingPathComponent("SmokeResult.json"), options: .atomic)
    } catch { print("Cannot save smoke result: \(error)") }
  }

  init() {
    do {
      guard let resources = Bundle.main.resourceURL?.appendingPathComponent("SparkiumResources")
      else {
        throw CocoaError(.fileNoSuchFile)
      }
      self.resources = resources
      scenes = try JSONDecoder().decode(
        [BundledScene].self,
        from: Data(contentsOf: resources.appendingPathComponent("catalog.json")))
      guard !scenes.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
      if !scenes.contains(where: { $0.id == selection }) { selection = scenes[0].id }
    } catch { self.error = "Cannot load bundled scenes: \(error.localizedDescription)" }
  }

  func render() {
    guard !busy, let resources else { return }
    busy = true
    stopping = false
    image = nil
    spp = 0
    elapsed = 0
    status = "Loading scene and preparing Metal…"
    renderer.render(
      resources: resources, scene: selection, dimension: dimension, samples: targetSamples,
      progress: { [weak self] image, spp, seconds, device in
        MainActor.assumeIsolated {
          guard let self else { return }
          self.image = image
          self.spp = spp
          self.elapsed = seconds
          self.backend = "Metal · \(device) · Path Tracing - Ray Query"
          self.status = self.stopping ? "Stopping after this sample…" : "Rendering"
        }
      },
      completion: { [weak self] error, cancelled in
        MainActor.assumeIsolated {
          guard let self else { return }
          self.busy = false
          self.stopping = false
          self.status = error != nil ? "Render failed" : (cancelled ? "Stopped" : "Complete")
          self.error = error
          self.saveSmokeResult()
        }
      })
  }

  func cancel() {
    guard busy else { return }
    stopping = true
    status = "Stopping after the current operation…"
    renderer.cancel()
  }
}

@main
struct SparkiumApp: App {
  @StateObject private var model = RenderModel()
  @Environment(\.scenePhase) private var phase
  var body: some Scene {
    WindowGroup {
      NavigationStack {
        Form {
          Section {
            Group {
              if let image = model.image {
                Image(decorative: image, scale: 1).resizable().scaledToFit()
              } else {
                RoundedRectangle(cornerRadius: 12).fill(.quaternary)
                  .overlay {
                    if model.busy {
                      ProgressView("Preparing scene…")
                    } else {
                      Label("Choose a scene to render", systemImage: "photo")
                    }
                  }
                  .frame(height: 220)
              }
            }.frame(maxWidth: .infinity, maxHeight: 300)
              .accessibilityLabel("Rendered scene")
            Text(model.backend).font(.caption).foregroundStyle(.secondary)
            HStack {
              Text("\(model.spp) / \(model.targetSamples) spp")
              Spacer()
              Text(model.elapsed, format: .number.precision(.fractionLength(1))) + Text(" s")
            }.monospacedDigit()
            ProgressView(value: Double(model.spp), total: Double(model.targetSamples))
            Text(model.status).font(.callout)
          }
          Section("Render settings") {
            Picker("Scene", selection: $model.selection) {
              ForEach(model.scenes) { Text($0.name).tag($0.id) }
            }
            Picker("Longest edge", selection: $model.dimension) {
              ForEach([128, 256, 512, 1024], id: \.self) { Text("\($0) px").tag($0) }
            }
            Picker("Samples", selection: $model.targetSamples) {
              ForEach([1, 2, 8, 32, 64, 128, 256], id: \.self) { Text("\($0) spp").tag($0) }
            }
          }.disabled(model.busy)
        }
        .navigationTitle("Sparkium")
        .toolbar {
          ToolbarItem(placement: .topBarTrailing) {
            if model.busy {
              Button("Stop", role: .cancel) { model.cancel() }.disabled(model.stopping)
            } else {
              Button("Render") { model.render() }.disabled(model.scenes.isEmpty)
            }
          }
        }
        .alert(
          "Sparkium",
          isPresented: Binding(
            get: { model.error != nil },
            set: { if !$0 { model.error = nil } })
        ) {
          Button("OK") { model.error = nil }
        } message: {
          Text(model.error ?? "")
        }
      }
      .onChange(of: phase) { _, phase in
        if phase == .active { model.startSmokeTestIfRequested() } else { model.cancel() }
      }
      .task { if phase == .active { model.startSmokeTestIfRequested() } }
    }
  }
}
