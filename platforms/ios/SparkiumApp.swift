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
  @Published var viewportWidth = 0
  @Published var viewportHeight = 0
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
    guard !smokeStarted, viewportWidth > 0, viewportHeight > 0,
      let scene = ProcessInfo.processInfo.environment["SPARKIUM_SMOKE_SCENE"],
      scenes.contains(where: { $0.id == scene })
    else { return }
    smokeStarted = true
    selection = scene
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
        "width": image?.width ?? 0, "height": image?.height ?? 0,
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
    guard !busy, viewportWidth > 0, viewportHeight > 0, let resources else { return }
    busy = true
    stopping = false
    image = nil
    spp = 0
    elapsed = 0
    status = "Loading scene and preparing Metal…"
    renderer.render(
      resources: resources, scene: selection, dimension: max(viewportWidth, viewportHeight),
      aspectRatio: Double(viewportWidth) / Double(viewportHeight), samples: targetSamples,
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

// The viewport always keeps the full window size; controls only overlay it.
private struct RenderViewport: View {
  @ObservedObject var model: RenderModel
  @State private var controlsVisible = true
  @Environment(\.displayScale) private var displayScale

  var body: some View {
    ZStack(alignment: .topLeading) {
      GeometryReader { viewport in
        ZStack {
          Color.black
          if let image = model.image {
            Image(decorative: image, scale: 1)
              .resizable()
              .scaledToFit()
              .frame(width: viewport.size.width, height: viewport.size.height)
              .accessibilityLabel("Rendered scene")
          } else if model.busy {
            ProgressView("Preparing scene…")
              .tint(.white)
          } else {
            Label("Open controls to choose a scene and render", systemImage: "photo")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
        }
        .frame(width: viewport.size.width, height: viewport.size.height)
        .onAppear { updateViewport(viewport.size) }
        .onChange(of: viewport.size) { _, size in updateViewport(size) }
        .onChange(of: displayScale) { _, _ in updateViewport(viewport.size) }
      }
      .ignoresSafeArea()

      GeometryReader { available in
        if controlsVisible {
          controls
            .frame(width: min(340, max(0, available.size.width - 20)))
            .frame(maxHeight: max(0, available.size.height - 20), alignment: .top)
            .padding(10)
            .transition(.move(edge: .leading).combined(with: .opacity))
        } else {
          Button {
            withAnimation(.easeInOut(duration: 0.2)) { controlsVisible = true }
          } label: {
            Image(systemName: "sidebar.left")
              .font(.title3)
              .frame(width: 44, height: 44)
          }
          .buttonStyle(.plain)
          .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
          .padding(10)
          .accessibilityLabel("Show render controls")
          .accessibilityIdentifier("show-render-controls")
        }
      }
    }
    .preferredColorScheme(.dark)
    .statusBarHidden()
    .persistentSystemOverlays(.hidden)
    .onChange(of: model.busy, initial: true) { _, busy in
      if busy { withAnimation(.easeInOut(duration: 0.2)) { controlsVisible = false } }
    }
    .onChange(of: model.error) { _, error in
      if error != nil { controlsVisible = true }
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

  private func updateViewport(_ size: CGSize) {
    if size.width > 0 && size.height > 0 {
      model.viewportWidth = Int((size.width * displayScale).rounded())
      model.viewportHeight = Int((size.height * displayScale).rounded())
      model.startSmokeTestIfRequested()
    }
  }

  private var controls: some View {
    VStack(spacing: 0) {
      HStack {
        Text("Sparkium scenes").font(.headline)
        Spacer()
        Button {
          withAnimation(.easeInOut(duration: 0.2)) { controlsVisible = false }
        } label: {
          Image(systemName: "xmark").frame(width: 44, height: 44)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Hide render controls")
        .accessibilityIdentifier("hide-render-controls")
      }
      .padding(.leading, 14)
      .padding(.trailing, 2)
      Divider()
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          VStack(spacing: 6) {
            LabeledContent("Scene") {
              Picker("Scene", selection: $model.selection) {
                ForEach(model.scenes) { Text($0.name).tag($0.id) }
              }.labelsHidden()
            }.frame(minHeight: 38)
            LabeledContent("Resolution") {
              Text("\(model.viewportWidth) × \(model.viewportHeight)")
                .monospacedDigit()
                .foregroundStyle(.secondary)
            }.frame(minHeight: 38)
            LabeledContent("Target spp") {
              Picker("Target spp", selection: $model.targetSamples) {
                ForEach([1, 2, 8, 32, 64, 128, 256], id: \.self) { Text("\($0) spp").tag($0) }
              }.labelsHidden()
            }.frame(minHeight: 38)
          }
          .pickerStyle(.menu)
          .disabled(model.busy)
          Divider()
          VStack(alignment: .leading, spacing: 6) {
            Text(model.backend)
            HStack {
              Text("Accumulated spp: \(model.spp)")
              Spacer()
              Text(model.elapsed, format: .number.precision(.fractionLength(1))) + Text(" s")
            }.monospacedDigit()
            ProgressView(value: Double(model.spp), total: Double(model.targetSamples))
            Text(model.status).foregroundStyle(.secondary)
          }
          .font(.caption)
        }
        .padding(14)
      }
      .scrollBounceBehavior(.basedOnSize)
      Divider()
      HStack {
        if model.busy {
          Button("Stop", role: .cancel) { model.cancel() }.disabled(model.stopping)
        } else {
          Button(model.image == nil ? "Render" : "Render again") { model.render() }
            .disabled(model.scenes.isEmpty)
        }
        Spacer()
        Text("\(model.targetSamples) spp").font(.caption).foregroundStyle(.secondary)
      }
      .buttonStyle(.bordered)
      .controlSize(.regular)
      .padding(.horizontal, 14)
      .padding(.vertical, 8)
    }
    .background(.black.opacity(0.75))
    .background(.ultraThinMaterial)
    .clipShape(RoundedRectangle(cornerRadius: 12))
    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.15), lineWidth: 1))
    .shadow(color: .black.opacity(0.4), radius: 16, x: 0, y: 6)
    .accessibilityIdentifier("render-controls")
  }
}

@main
struct SparkiumApp: App {
  @StateObject private var model = RenderModel()
  @Environment(\.scenePhase) private var phase
  var body: some Scene {
    WindowGroup {
      RenderViewport(model: model)
        .onChange(of: phase) { _, phase in
          if phase == .active { model.startSmokeTestIfRequested() } else { model.cancel() }
        }
        .task { if phase == .active { model.startSmokeTestIfRequested() } }
    }
  }
}
