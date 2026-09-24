import MetalKit
import SwiftUI
import UniformTypeIdentifiers

private final class GameMetalView: MTKView {
  let renderer = DemoRenderer()
  var life = false
  private var start = CGPoint.zero
  private var pointerDown = false
  override init(frame: CGRect, device: MTLDevice?) {
    super.init(frame: frame, device: device)
    isMultipleTouchEnabled = true
    let pinch = UIPinchGestureRecognizer(target: self, action: #selector(magnify(_:)))
    addGestureRecognizer(pinch)
    let pan = UIPanGestureRecognizer(target: self, action: #selector(pan(_:)))
    pan.minimumNumberOfTouches = 2
    pan.maximumNumberOfTouches = 2
    addGestureRecognizer(pan)
  }
  required init(coder: NSCoder) { fatalError() }
  private func send(_ kind: Int, _ point: CGPoint, _ value: Double = 0) {
    guard bounds.width > 0 && bounds.height > 0 else { return }
    renderer.input(kind, x: point.x / bounds.width, y: point.y / bounds.height, value: value)
  }
  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
    guard event?.allTouches?.count == 1, let touch = touches.first else {
      cancelPointer()
      return
    }
    start = touch.location(in: self)
    pointerDown = true
    send(1, start)
  }
  override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
    if pointerDown, let touch = touches.first { send(0, touch.location(in: self)) }
  }
  override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
    guard pointerDown, let touch = touches.first else { return }
    let end = touch.location(in: self)
    send(2, end)
    pointerDown = false
    let dx = end.x - start.x
    let dy = end.y - start.y
    if !life && max(abs(dx), abs(dy)) > 24 {
      // GLFW key values are the existing Window input vocabulary, not GLFW calls.
      let key = abs(dx) > abs(dy) ? (dx > 0 ? 262 : 263) : (dy > 0 ? 264 : 265)
      renderer.input(4, x: 0, y: 0, value: Double(key))
    }
  }
  override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { cancelPointer() }
  fileprivate func cancelPointer(force: Bool = false) {
    if pointerDown || force {
      renderer.input(5, x: 0, y: 0, value: 0)
      renderer.input(5, x: 0, y: 0, value: 1)
      pointerDown = false
    }
  }
  @objc private func magnify(_ gesture: UIPinchGestureRecognizer) {
    guard life else { return }
    cancelPointer()
    send(6, gesture.location(in: self), gesture.scale)
    gesture.scale = 1
  }
  @objc private func pan(_ gesture: UIPanGestureRecognizer) {
    guard life else { return }
    cancelPointer()
    let point = gesture.location(in: self)
    if gesture.state == .began {
      send(1, point, 1)
    } else if gesture.state == .changed {
      send(0, point)
    } else {
      send(2, point, 1)
    }
  }
}

// Keep Life's canvas in the phone's physical portrait coordinates. UIKit maps
// touches into the transformed Metal view, including pinch and two-finger pan.
private final class GameCanvasView: UIView {
  let metal = GameMetalView(frame: .zero, device: nil)
  private var canvasAngle: CGFloat?
  override init(frame: CGRect) {
    super.init(frame: frame)
    addSubview(metal)
  }
  required init?(coder: NSCoder) { fatalError() }
  override func layoutSubviews() {
    super.layoutSubviews()
    var angle: CGFloat = 0
    if metal.life {
      switch window?.windowScene?.interfaceOrientation {
      case .landscapeLeft: angle = -.pi / 2
      case .landscapeRight: angle = .pi / 2
      case .portraitUpsideDown: angle = .pi
      default: break
      }
    }
    if canvasAngle != angle {
      metal.cancelPointer(force: true)
      metal.renderer.setGameIconRotation(Float(-angle))
      canvasAngle = angle
    }
    let sideways = abs(sin(angle)) > 0.5
    metal.bounds = CGRect(
      origin: .zero,
      size: sideways
        ? CGSize(width: bounds.height, height: bounds.width) : bounds.size)
    metal.center = CGPoint(x: bounds.midX, y: bounds.midY)
    metal.transform = CGAffineTransform(rotationAngle: angle)
  }
  override func didMoveToWindow() {
    super.didMoveToWindow()
    setNeedsLayout()
    // Exercise real scene rotation in headless simulator smoke runs.
    if let direction = ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_ORIENTATION"],
      let scene = window?.windowScene
    {
      let orientation: UIInterfaceOrientationMask =
        direction == "left"
        ? .landscapeLeft
        : direction == "right" ? .landscapeRight : .portrait
      DispatchQueue.main.async {
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: orientation))
      }
    }
  }
}

private struct DesktopGameView: UIViewRepresentable {
  let life: Bool
  let active: Bool
  var status: (Double, String?) -> Void
  final class Coordinator: NSObject, UIDocumentPickerDelegate {
    weak var view: GameMetalView?
    var error: (String?) -> Void = { _ in }
    private var exporting = false
    private var temporary: URL?
    func request(_ action: Int) {
      guard let view, let root = view.window?.rootViewController else { return }
      if action == 2 {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Life.cells")
        temporary = url
        view.renderer.completeFile(url.path) { [weak self] message in
          guard let self else { return }
          if let message {
            self.error(message)
            return
          }
          self.exporting = true
          let picker = UIDocumentPickerViewController(forExporting: [url], asCopy: true)
          picker.delegate = self
          root.present(picker, animated: true)
        }
      } else {
        exporting = false
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.item], asCopy: true)
        picker.delegate = self
        root.present(picker, animated: true)
      }
    }
    func documentPicker(
      _ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]
    ) {
      if exporting {
        cleanup()
        return
      }
      guard let url = urls.first else {
        cancel()
        return
      }
      let scoped = url.startAccessingSecurityScopedResource()
      view?.renderer.completeFile(url.path) { [weak self] message in
        if scoped { url.stopAccessingSecurityScopedResource() }
        self?.error(message)
      }
    }
    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
      if !exporting { cancel() }
      cleanup()
    }
    private func cancel() { view?.renderer.completeFile("") { _ in } }
    private func cleanup() {
      if let temporary { try? FileManager.default.removeItem(at: temporary) }
      temporary = nil
    }
  }
  func makeCoordinator() -> Coordinator { Coordinator() }
  func makeUIView(context: Context) -> GameCanvasView {
    let canvas = GameCanvasView(frame: .zero)
    let view = canvas.metal
    view.life = life
    context.coordinator.view = view
    context.coordinator.error = { status(0, $0) }
    view.renderer.fileRequest = { [weak coordinator = context.coordinator] action in
      coordinator?.request(action)
    }
    let resources = Bundle.main.resourceURL!.appendingPathComponent("SparkiumResources")
    view.renderer.start(view: view, resources: resources, demo: life ? "gol" : "2048") {
      _, _, fps, _, _, _, _, error in status(fps, error)
    }
    return canvas
  }
  func updateUIView(_ view: GameCanvasView, context: Context) {
    view.metal.renderer.setActive(active)
  }
  static func dismantleUIView(_ view: GameCanvasView, coordinator: Coordinator) {
    view.metal.renderer.stop()
  }
}
struct GamesView: View {
  let life: Bool
  let onExit: () -> Void
  @Environment(\.scenePhase) private var phase
  @State private var fps = 0.0
  @State private var error: String?
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Text("[Metal] \(life ? "Game of Life" : "2048") FPS: \(fps,specifier:"%.1f")").font(
          .caption.monospaced())
        Spacer()
        Button("Demos", action: onExit)
      }.padding(.horizontal, 12).frame(height: 36)
      DesktopGameView(life: life, active: phase == .active) { value, message in
        fps = value
        error = message
      }
    }.background(.black)
      .alert(
        "Game error", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })
      ) {
        Button("OK") { error = nil }
      } message: {
        Text(error ?? "")
      }
  }
}
