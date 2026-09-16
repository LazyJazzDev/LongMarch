import ImageIO
import SwiftUI
import UIKit

private struct BundledScene: Decodable, Identifiable {
  let id: String
  let name: String
  let width: Int?
  let height: Int?
  var displayName: String {
    if id.hasPrefix("blender_") {
      return "Blender "
        + id.dropFirst("blender_".count).replacingOccurrences(of: "_", with: " ").capitalized
    }
    return name
  }
}

@MainActor
private final class RenderModel: ObservableObject {
  @Published var scenes: [BundledScene] = []
  @Published var selection = "cornell_box" {
    didSet { if started && selection != oldValue { loadSelectedScene() } }
  }
  @Published var targetSamples = 32 {
    didSet {
      if started && error == nil {
        renderer.setSampleLimit(targetSamples)
        updateStatus()
      }
    }
  }
  @Published var image: CGImage?
  @Published var imageRevision = 0
  @Published var width = 0
  @Published var height = 0
  @Published var spp = 0
  @Published var elapsed = 0.0
  @Published var fps = 0.0
  @Published var raysPerSecond = 0.0
  @Published var device = "—"
  @Published var maxBounces = 0
  @Published var status = "Loading…"
  @Published var busy = false
  @Published var loaded = false
  @Published var error: String?
  private let renderer = SparkiumRenderer()
  private var resources: URL?
  private var started = false
  private var active = false
  private var request = 0
  private var smoke = false

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
      if let scene = ProcessInfo.processInfo.environment["SPARKIUM_SMOKE_SCENE"],
        scenes.contains(where: { $0.id == scene })
      {
        smoke = true
        selection = scene
        targetSamples = 2
      }
    } catch { self.error = "Cannot load bundled scenes: \(error.localizedDescription)" }
  }

  func setActive(_ value: Bool) {
    guard active != value || !started else { return }
    active = value
    renderer.setPaused(!value)
    if value && !started {
      started = true
      loadSelectedScene()
    } else {
      updateStatus()
    }
  }

  func loadSelectedScene() {
    guard let resources, scenes.contains(where: { $0.id == selection }) else { return }
    request += 1
    let current = request
    imageRevision += 1
    image = nil
    spp = 0
    elapsed = 0
    fps = 0
    raysPerSecond = 0
    loaded = false
    error = nil
    device = "—"
    maxBounces = 0
    width = scenes.first(where: { $0.id == selection })?.width ?? 0
    height = scenes.first(where: { $0.id == selection })?.height ?? 0
    updateStatus()
    renderer.load(
      resources: resources, scene: selection, samples: targetSamples,
      progress: { [weak self] image, spp, seconds, frameSeconds, device, width, height, bounces in
        MainActor.assumeIsolated {
          guard let self, self.request == current else { return }
          self.loaded = true
          if let image { self.image = image }
          self.spp = spp
          self.elapsed = seconds
          self.device = device
          self.width = width
          self.height = height
          self.maxBounces = bounces
          self.fps = frameSeconds > 0 ? 1.0 / frameSeconds : 0
          // One camera ray per pixel, one sample per rendered frame, just like the desktop readout.
          self.raysPerSecond = Double(width) * Double(height) * self.fps
          self.updateStatus()
        }
      },
      completion: { [weak self] error, _ in
        MainActor.assumeIsolated {
          guard let self, self.request == current else { return }
          self.error = error
          self.updateStatus()
          if !self.busy { self.saveSmokeResult() }
        }
      })
  }

  func resetFilm() {
    guard loaded, error == nil else { return }
    spp = 0
    elapsed = 0
    fps = 0
    raysPerSecond = 0
    updateStatus()
    renderer.resetFilm()
  }

  private func updateStatus() {
    if error != nil {
      busy = false
      status = "Render failed"
    } else if !active {
      busy = false
      status = "Paused in background"
    } else if !loaded {
      busy = true
      status = "Loading scene and preparing Metal…"
    } else if spp >= targetSamples {
      busy = false
      status = "Sample limit reached"
    } else {
      busy = true
      status = "Rendering"
    }
  }

  private func saveSmokeResult() {
    guard smoke else { return }
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
        "error": error ?? "", "device": device, "seconds": elapsed, "width": width,
        "height": height,
        "fps": fps, "camera_rays_per_second": raysPerSecond, "sample_limit": targetSamples,
      ]
      try JSONSerialization.data(withJSONObject: result, options: .prettyPrinted)
        .write(to: directory.appendingPathComponent("SmokeResult.json"), options: .atomic)
    } catch { print("Cannot save smoke result: \(error)") }
  }
}

// UIKit's native scroll/zoom gestures keep the point between the fingers anchored during a pinch.
// Replacing progressive image pixels never changes the zoom, offset, or render camera.
private final class RenderImageScrollView: UIScrollView, UIScrollViewDelegate {
  let renderedImage = UIImageView()
  private var revision = -1
  private var previousBounds = CGSize.zero
  private var needsFit = true

  override init(frame: CGRect) {
    super.init(frame: frame)
    delegate = self
    backgroundColor = .black
    contentInsetAdjustmentBehavior = .never
    showsHorizontalScrollIndicator = false
    showsVerticalScrollIndicator = false
    bouncesZoom = true
    addSubview(renderedImage)
    let doubleTap = UITapGestureRecognizer(target: self, action: #selector(fitImage))
    doubleTap.numberOfTapsRequired = 2
    addGestureRecognizer(doubleTap)
    accessibilityLabel = "Rendered scene"
    accessibilityHint = "Pinch to zoom, drag to pan, double-tap to fit the screen"
  }
  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

  func update(_ image: CGImage, revision: Int) {
    let imageSize = CGSize(width: image.width, height: image.height)
    let reset = self.revision != revision || renderedImage.bounds.size != imageSize
    renderedImage.image = UIImage(cgImage: image)
    if reset {
      self.revision = revision
      setZoomScale(1, animated: false)
      renderedImage.frame = CGRect(origin: .zero, size: imageSize)
      contentSize = imageSize
      needsFit = true
      setNeedsLayout()
    }
  }
  override func layoutSubviews() {
    super.layoutSubviews()
    if needsFit || previousBounds != bounds.size {
      previousBounds = bounds.size
      needsFit = false
      fitImage()
    }
    centerImage()
  }
  @objc func fitImage() {
    let size = renderedImage.bounds.size
    guard size.width > 0, size.height > 0, bounds.width > 0, bounds.height > 0 else { return }
    let fit = min(bounds.width / size.width, bounds.height / size.height)
    minimumZoomScale = fit
    maximumZoomScale = max(1, fit * 16)
    setZoomScale(fit, animated: false)
    centerImage()
    contentOffset = CGPoint(x: -contentInset.left, y: -contentInset.top)
  }
  private func centerImage() {
    let horizontal = max(0, (bounds.width - renderedImage.frame.width) / 2)
    let vertical = max(0, (bounds.height - renderedImage.frame.height) / 2)
    let inset = UIEdgeInsets(top: vertical, left: horizontal, bottom: vertical, right: horizontal)
    if contentInset != inset { contentInset = inset }
  }
  func viewForZooming(in scrollView: UIScrollView) -> UIView? { renderedImage }
  func scrollViewDidZoom(_ scrollView: UIScrollView) { centerImage() }
}

private struct ZoomableRenderImage: UIViewRepresentable {
  let image: CGImage
  let revision: Int
  func makeUIView(context: Context) -> RenderImageScrollView { RenderImageScrollView() }
  func updateUIView(_ view: RenderImageScrollView, context: Context) {
    view.update(image, revision: revision)
  }
}

private struct RenderViewport: View {
  @ObservedObject var model: RenderModel
  @State private var controlsVisible = true
  var body: some View {
    ZStack(alignment: .topLeading) {
      ZStack {
        Color.black
        if let image = model.image {
          ZoomableRenderImage(image: image, revision: model.imageRevision)
        } else if model.busy {
          ProgressView("Preparing scene…").tint(.white)
        } else if model.error != nil {
          Text("Select another scene or reload").foregroundStyle(.secondary)
        }
      }
      .ignoresSafeArea()

      GeometryReader { available in
        if controlsVisible {
          controls
            .frame(width: min(350, max(0, available.size.width - 20)))
            .frame(maxHeight: max(0, available.size.height - 20), alignment: .top)
            .padding(10)
            .transition(.move(edge: .leading).combined(with: .opacity))
        } else {
          Button {
            withAnimation(.easeInOut(duration: 0.2)) { controlsVisible = true }
          } label: {
            Image(systemName: "sidebar.left").font(.title3).frame(width: 44, height: 44)
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
    .onChange(of: model.error) { _, error in if error != nil { controlsVisible = true } }
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
      }.padding(.leading, 14).padding(.trailing, 2)
      Divider()
      ScrollView {
        VStack(alignment: .leading, spacing: 10) {
          LabeledContent("Scene") {
            Picker("Scene", selection: $model.selection) {
              ForEach(model.scenes) { Text($0.displayName).tag($0.id) }
            }.labelsHidden().accessibilityIdentifier("scene-picker")
          }.frame(minHeight: 38)
          LabeledContent("SPP limit") {
            Picker("SPP limit", selection: $model.targetSamples) {
              ForEach([1, 2, 8, 32, 64, 128, 256, 512, 1024, 4096], id: \.self) {
                Text("\($0)").tag($0)
              }
            }.labelsHidden().accessibilityIdentifier("spp-limit")
          }.frame(minHeight: 38)
          HStack {
            Button("Reload") { model.loadSelectedScene() }
            Button("Reset film") { model.resetFilm() }.disabled(!model.loaded || model.error != nil)
          }.buttonStyle(.bordered)
          Divider()
          VStack(alignment: .leading, spacing: 6) {
            HStack {
              Text("Ray/s: \(model.raysPerSecond / 1e6, specifier: "%.2f") M")
                .accessibilityHint("Last frame: image width × height × samples per frame × FPS")
              Spacer()
              Text("FPS: \(model.fps, specifier: "%.1f")")
            }
            Text("Accumulated spp: \(model.spp)").accessibilityIdentifier("accumulated-spp")
            ProgressView(
              value: Double(min(model.spp, model.targetSamples)), total: Double(model.targetSamples)
            )
            Text(model.status).foregroundStyle(.secondary)
            Text("Render time: \(model.elapsed, specifier: "%.1f") s")
            Text("Backend: Metal")
            Text("Device: \(model.device)")
            Text("Pipeline: Path Tracing - Ray Query")
            Text(
              model.width > 0
                ? "Resolution: \(model.width) × \(model.height)" : "Resolution: loading…")
            Text("Samples / frame: 1 · Max bounces: \(model.maxBounces)")
            Text("Ray/s and FPS describe the last rendered frame.").foregroundStyle(.secondary)
            Text("scenes/\(model.selection)/scene.json").foregroundStyle(.secondary)
            if let error = model.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
          }.font(.caption).monospacedDigit()
        }.padding(14)
      }
      .pickerStyle(.menu)
      .scrollBounceBehavior(.basedOnSize)
      Divider()
      Text("Pinch to zoom · Drag to pan · Double-tap to fit")
        .font(.caption2).foregroundStyle(.secondary).padding(12)
    }
    .background(.black.opacity(0.75)).background(.ultraThinMaterial)
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
        .onChange(of: phase) { _, phase in model.setActive(phase == .active) }
        .task { model.setActive(phase == .active) }
    }
  }
}
