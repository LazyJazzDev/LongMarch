import MetalKit
import SwiftUI

@MainActor
private final class DemoModel: ObservableObject {
  @Published var frameSeconds = 0.0
  @Published var fps = 0.0
  @Published var gpuMilliseconds = 0.0
  @Published var device = "—"
  @Published var width = 0
  @Published var height = 0
  @Published var frames = 0
  @Published var error: String?
  @Published var particles = 4096
  @Published var galaxies = 10
  @Published var deltaTime = 0.03
  @Published var simulate = true
  @Published var reset = 0
  @Published var yaw = 0.0
  @Published var pitch = 0.0
  @Published var resolutionScale = 1.0
}

private struct MetalDemoView: UIViewRepresentable {
  let demo: DemoItem
  @ObservedObject var model: DemoModel
  let active: Bool
  final class Coordinator {
    let renderer = DemoRenderer()
  }
  func makeCoordinator() -> Coordinator { Coordinator() }
  func makeUIView(context: Context) -> MTKView {
    let view = MTKView(frame: .zero)
    let resources = Bundle.main.resourceURL!.appendingPathComponent("SparkiumResources")
    context.coordinator.renderer.start(view: view, resources: resources, demo: demo.id) {
      seconds, gpuMS, fps, device, width, height, frames, error in
      MainActor.assumeIsolated {
        model.fps = fps
        model.frameSeconds = seconds
        model.gpuMilliseconds = gpuMS
        model.device = device
        model.width = width
        model.height = height
        model.frames = frames
        model.error = error
      }
    }
    return view
  }
  func updateUIView(_ view: MTKView, context: Context) {
    context.coordinator.renderer.configure(
      particles: model.particles, galaxies: model.galaxies,
      deltaTime: Float(model.deltaTime), simulate: model.simulate, yaw: Float(model.yaw),
      pitch: Float(model.pitch),
      reset: model.reset, resolutionScale: Float(model.resolutionScale))
    context.coordinator.renderer.setActive(active)
  }
  static func dismantleUIView(_ view: MTKView, coordinator: Coordinator) {
    coordinator.renderer.stop()
  }
}

struct GraphicsDemoView: View {
  let demo: DemoItem
  let onExit: () -> Void
  @StateObject private var model = DemoModel()
  @Environment(\.scenePhase) private var phase
  @State private var controlsVisible = true
  @State private var dragStart = CGSize.zero
  private var nbody: Bool { demo.id == "nbody_cs" }
  private var resizable: Bool { nbody || demo.id == "graphics_hello_resize" }
  var body: some View {
    ZStack(alignment: .topLeading) {
      MetalDemoView(demo: demo, model: model, active: phase == .active).ignoresSafeArea()
        .gesture(
          DragGesture().onChanged { value in
            guard nbody else { return }
            model.yaw = dragStart.width + value.translation.width * .pi / 180
            model.pitch = dragStart.height + value.translation.height * .pi / 180
          }.onEnded { _ in dragStart = CGSize(width: model.yaw, height: model.pitch) })
      GeometryReader { geometry in
        if controlsVisible {
          controls.frame(width: min(350, geometry.size.width - 20))
            .frame(maxHeight: geometry.size.height - 20, alignment: .top).padding(10)
        } else {
          Button {
            controlsVisible = true
          } label: {
            Image(systemName: "sidebar.left").frame(width: 44, height: 44)
          }
          .buttonStyle(.bordered).padding(10)
        }
      }
      HStack {
        Spacer()
        Button(action: onExit) { Label("Demos", systemImage: "list.bullet") }
          .buttonStyle(.borderedProminent)
      }.padding(12)
    }
  }
  private var controls: some View {
    VStack(spacing: 0) {
      HStack {
        Text(demo.title).font(.headline)
        Spacer()
        Button {
          controlsVisible = false
        } label: {
          Image(systemName: "xmark").frame(width: 36, height: 36)
        }
        .accessibilityLabel("Hide demo controls")
      }.padding(.horizontal, 14)
      Divider()
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          Text(demo.subtitle).font(.subheadline).foregroundStyle(.secondary)
          if nbody {
            LabeledContent("Particles") {
              Picker("Particles", selection: $model.particles) {
                ForEach([1024, 2048, 4096, 8192, 16384, 32768, 65536], id: \.self) {
                  Text("\($0)").tag($0)
                }
              }.labelsHidden()
            }
            Stepper("Galaxies: \(model.galaxies)", value: $model.galaxies, in: 1...20)
            Text("Time step: \(model.deltaTime, specifier: "%.3f")")
            Slider(value: $model.deltaTime, in: 0.001...0.1)
            Button("Reset particles") { model.reset += 1 }
            Text("Drag the image to rotate the camera.").font(.caption).foregroundStyle(.secondary)
          }
          if resizable {
            Toggle(nbody ? "Simulate" : "Animate", isOn: $model.simulate)
            LabeledContent("Render scale") {
              Picker("Render scale", selection: $model.resolutionScale) {
                Text("25%").tag(0.25)
                Text("50%").tag(0.5)
                Text("100%").tag(1.0)
              }.labelsHidden()
            }
          }
          Divider()
          VStack(alignment: .leading, spacing: 6) {
            Text(model.frames == 0 ? "Preparing Metal…" : "Frame: \(model.frames)")
            Text(
              "FPS: \(model.fps, specifier: "%.1f") · Render: \(model.frameSeconds * 1000, specifier: "%.2f") ms"
            )
            Text("GPU: \(model.gpuMilliseconds, specifier: "%.2f") ms")
            if nbody && model.simulate {
              Text(
                "Estimated: \(model.frameSeconds > 0 ? Double(model.particles * model.particles) * 20 / model.frameSeconds / 1e9 : 0, specifier: "%.1f") GFLOP/s"
              )
            }
            Text("Resolution: \(model.width) × \(model.height)")
            Text("Backend: Metal")
            Text("Device: \(model.device)")
            Text("Pipeline: \(nbody ? "Compute + Rasterization" : "Rasterization")")
            Text("Display refresh is capped at 60 Hz.")
              .foregroundStyle(.secondary)
            if let error = model.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
          }.font(.caption).monospacedDigit()
        }.padding(14)
      }.pickerStyle(.menu)
    }
    .background(.black.opacity(0.75)).background(.ultraThinMaterial)
    .clipShape(RoundedRectangle(cornerRadius: 12))
    .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.15)))
  }
}
