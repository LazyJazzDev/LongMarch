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
  private func cancelPointer() {
    if pointerDown {
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

// The window stays in its entry orientation. Only the icon angle follows the
// device: there is no counter-rotated rectangular view or relayout to animate.
private final class GameCanvasView: UIView {
  let metal = GameMetalView(frame: .zero, device: nil)
  private weak var lockedScene: UIWindowScene?
  private var entryOrientation: UIInterfaceOrientation = .portrait
  private var iconAngle: CGFloat?

  override init(frame: CGRect) {
    super.init(frame: frame)
    addSubview(metal)
  }
  required init?(coder: NSCoder) { fatalError() }

  override func layoutSubviews() {
    super.layoutSubviews()
    metal.frame = bounds
  }

  override func didMoveToWindow() {
    super.didMoveToWindow()
    if window == nil {
      stopOrientationTracking()
    } else if metal.life, lockedScene == nil, let scene = window?.windowScene {
      entryOrientation = scene.interfaceOrientation
      if entryOrientation == .unknown { entryOrientation = .portrait }
      lockedScene = scene
      DemoAppDelegate.orientationMask = mask(for: entryOrientation)
      scene.windows.first(where: { $0.isKeyWindow })?.rootViewController?
        .setNeedsUpdateOfSupportedInterfaceOrientations()
      UIDevice.current.beginGeneratingDeviceOrientationNotifications()
      NotificationCenter.default.addObserver(
        self, selector: #selector(deviceDidRotate),
        name: UIDevice.orientationDidChangeNotification, object: nil)
      deviceDidRotate()
      if ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_ORIENTATION"] != nil {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { [weak self] in
          self?.checkOrientationLock()
        }
      }
    }
  }

  func stopOrientationTracking() {
    guard let scene = lockedScene else { return }
    NotificationCenter.default.removeObserver(
      self, name: UIDevice.orientationDidChangeNotification, object: nil)
    UIDevice.current.endGeneratingDeviceOrientationNotifications()
    lockedScene = nil
    iconAngle = nil
    DemoAppDelegate.orientationMask = .allButUpsideDown
    scene.windows.first(where: { $0.isKeyWindow })?.rootViewController?
      .setNeedsUpdateOfSupportedInterfaceOrientations()
  }

  private func mask(for orientation: UIInterfaceOrientation) -> UIInterfaceOrientationMask {
    switch orientation {
    case .landscapeLeft: return .landscapeLeft
    case .landscapeRight: return .landscapeRight
    case .portraitUpsideDown: return .portraitUpsideDown
    default: return .portrait
    }
  }

  private func angle(for orientation: UIInterfaceOrientation) -> CGFloat {
    switch orientation {
    case .landscapeLeft: return .pi / 2
    case .landscapeRight: return -.pi / 2
    case .portraitUpsideDown: return .pi
    default: return 0
    }
  }

  @objc func deviceDidRotate() {
    // Supply sensor directions in headless smoke runs; production uses the
    // real device notification even while interface rotation is locked.
    switch ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_ORIENTATION"] {
    case "left": updateIcons(for: .landscapeLeft)
    case "right", "exit": updateIcons(for: .landscapeRight)
    case "portrait": updateIcons(for: .portrait)
    default: updateIcons(for: UIDevice.current.orientation)
    }
  }

  private func updateIcons(for deviceOrientation: UIDeviceOrientation) {
    guard lockedScene != nil else { return }
    let upright: UIInterfaceOrientation
    switch deviceOrientation {
    case .portrait: upright = .portrait
    case .portraitUpsideDown: upright = .portraitUpsideDown
    case .landscapeLeft: upright = .landscapeRight
    case .landscapeRight: upright = .landscapeLeft
    default: return  // Flat/unknown devices keep the last useful icon orientation.
    }
    // Framebuffer coordinates point down: compensate the device turn instead
    // of applying the window-orientation transform to the icon a second time.
    let radians = angle(for: entryOrientation) - angle(for: upright)
    guard iconAngle != radians else { return }
    iconAngle = radians
    metal.renderer.setGameIconRotation(Float(radians))
  }

  // Verify the actual UIWindowScene refuses rotation, not just that the final
  // framebuffer looks upright after a system rotation animation.
  private func checkOrientationLock() {
    guard let scene = lockedScene else { return }
    let before = scene.interfaceOrientation
    let beforeBounds = bounds
    let requested: UIInterfaceOrientationMask = before == .portrait ? .landscapeRight : .portrait
    var rejected = false
    scene.requestGeometryUpdate(.iOS(interfaceOrientations: requested)) { _ in rejected = true }
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
      guard let self, self.lockedScene === scene else { return }
      let result: [String: Any] = [
        "rotation_rejected": rejected,
        "interface_unchanged": scene.interfaceOrientation == before,
        "bounds_unchanged": self.bounds == beforeBounds,
        "view_transform_identity": self.metal.transform.isIdentity,
        "canvas_fills_window": self.window.map {
          self.convert(self.bounds, to: $0).integral == $0.bounds.integral
        } ?? false,
        "icon_angle": self.iconAngle ?? 0,
      ]
      let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("OrientationSmoke.json")
      try? JSONSerialization.data(withJSONObject: result).write(to: url)
      if ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_ORIENTATION"] == "exit" {
        self.stopOrientationTracking()
        scene.requestGeometryUpdate(.iOS(interfaceOrientations: requested))
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
          let restored = ["rotation_restored": scene.interfaceOrientation != before]
          let exitURL = url.deletingLastPathComponent().appendingPathComponent(
            "OrientationExitSmoke.json")
          try? JSONSerialization.data(withJSONObject: restored).write(to: exitURL)
        }
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
    if active { view.deviceDidRotate() }
  }
  static func dismantleUIView(_ view: GameCanvasView, coordinator: Coordinator) {
    view.stopOrientationTracking()
    view.metal.renderer.stop()
  }
}
struct GamesView: View {
  let life: Bool
  let onExit: () -> Void
  @Environment(\.scenePhase) private var phase
  @State private var fps = 0.0
  @State private var error: String?
  private var game: some View {
    DesktopGameView(life: life, active: phase == .active) { value, message in
      fps = value
      error = message
    }
  }
  var body: some View {
    Group {
      if life {
        ZStack(alignment: .top) {
          game.ignoresSafeArea()
          // Only this small overlay uses the safe area. The Metal canvas fills
          // the display, with reset/random beside the island and controls below.
          HStack(spacing: 12) {
            Text("[Metal] GoL FPS: \(fps, specifier: "%.1f")")
              .font(.caption2.monospaced())
            Button("Demos", action: onExit).font(.caption)
          }
          .padding(.horizontal, 12).padding(.vertical, 6)
          .background(.black.opacity(0.45), in: Capsule())
          .padding(.top, 4)
        }
      } else {
        VStack(spacing: 0) {
          HStack {
            Text("[Metal] 2048 FPS: \(fps,specifier:"%.1f")").font(.caption.monospaced())
            Spacer()
            Button("Demos", action: onExit)
          }.padding(.horizontal, 12).frame(height: 36)
          game
        }
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
