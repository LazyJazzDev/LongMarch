import MetalKit
import SwiftUI
import UniformTypeIdentifiers

private final class GameMetalView: MTKView {
  let renderer = DemoRenderer()
  var life = false
  private var start = CGPoint.zero
  private(set) var lastPointerPosition = CGPoint.zero
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
  func smokeSizeTap(_ axis: Int) {
    guard life, bounds.height > bounds.width else { return }
    // Exercise the normal input -> readout -> native popover path in portrait.
    let bottom = max(bounds.width * 0.05, (window?.safeAreaInsets.bottom ?? 0) + 12)
    let point = CGPoint(
      x: bounds.width * 0.435,
      y: bounds.height - bottom - bounds.width * (axis == 1 ? 0.07875 : 0.02125))
    lastPointerPosition = point
    send(1, point)
    send(2, point)
  }

  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
    guard event?.allTouches?.count == 1, let touch = touches.first else {
      cancelPointer()
      return
    }
    start = touch.location(in: self)
    lastPointerPosition = start
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
    puzzleSwipe(from: start, to: end)
  }
  func puzzleSwipe(from start: CGPoint, to end: CGPoint) {
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
  private var blankSwipeStart: CGPoint?
  private weak var lockedScene: UIWindowScene?
  private var entryOrientation: UIInterfaceOrientation = .portrait
  private var iconAngle: CGFloat?
  private var bottomControlInset: CGFloat?

  override init(frame: CGRect) {
    super.init(frame: frame)
    isMultipleTouchEnabled = true
    addSubview(metal)
  }
  required init?(coder: NSCoder) { fatalError() }

  override func layoutSubviews() {
    super.layoutSubviews()
    metal.frame = bounds
    if !metal.life {
      metal.frame.size.height = min(bounds.height, bounds.width * (118.5 / 88))
    }
    if metal.life, let window, bounds.height > 0 {
      // Read the window inset: this view deliberately ignores SwiftUI safe
      // areas so its own inset can be zero. Keep an extra finger-sized gap.
      let safeBottom = convert(
        CGPoint(x: window.bounds.midX, y: window.bounds.maxY - window.safeAreaInsets.bottom),
        from: window
      ).y
      let homeInset = max(0, bounds.maxY - safeBottom)
      let fraction = homeInset > 0 ? (homeInset + 12) / bounds.height : 0
      if bottomControlInset != fraction {
        bottomControlInset = fraction
        metal.renderer.setGameBottomControlInset(Float(fraction))
      }
    }
  }

  // Touches that begin below the Metal canvas belong to this container. Reuse
  // the board's swipe threshold without sending clicks to its menu or score.
  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
    guard !metal.life, event?.allTouches?.count == 1, let touch = touches.first else {
      blankSwipeStart = nil
      return
    }
    blankSwipeStart = touch.location(in: self)
  }
  override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
    if let start = blankSwipeStart, let touch = touches.first {
      metal.puzzleSwipe(from: start, to: touch.location(in: self))
    }
    blankSwipeStart = nil
  }
  override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
    blankSwipeStart = nil
  }

  override func safeAreaInsetsDidChange() {
    super.safeAreaInsetsDidChange()
    setNeedsLayout()
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
      var gestureController = self.window?.rootViewController
      while let child = gestureController?.childForScreenEdgesDeferringSystemGestures {
        gestureController = child
      }
      let result: [String: Any] = [
        "defers_bottom_gestures": gestureController?.preferredScreenEdgesDeferringSystemGestures
          .contains(.bottom) ?? false,
        "rotation_rejected": rejected,
        "interface_unchanged": scene.interfaceOrientation == before,
        "bounds_unchanged": self.bounds == beforeBounds,
        "view_transform_identity": self.metal.transform.isIdentity,
        "canvas_fills_window": self.window.map {
          self.convert(self.bounds, to: $0).integral == $0.bounds.integral
        } ?? false,
        "icon_angle": self.iconAngle ?? 0,
        "bottom_control_margin_points": (self.bottomControlInset ?? 0) * self.bounds.height,
        "window_bottom_safe_area_points": self.window?.safeAreaInsets.bottom ?? 0,
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

private struct GridDimensionPicker: View {
  let axis: Int
  @State var value: Double
  let apply: (Int) -> Void
  let close: () -> Void

  var body: some View {
    VStack(spacing: 16) {
      HStack {
        Text(axis == 1 ? LocalizedStringKey("Grid width") : "Grid height").font(.headline)
        Spacer()
        Button("Done") {
          apply(Int(value))
          close()
        }
      }
      HStack {
        Text("\(Int(value))").font(.title2.monospacedDigit())
        Spacer()
        Stepper(
          "Adjust by one cell",
          value: Binding(
            get: { Int(value) },
            set: {
              value = Double($0)
              apply($0)
            }),
          in: 2...256
        ).labelsHidden()
      }
      Slider(value: $value, in: 2...256, step: 1) { editing in
        // Keep the native thumb responsive; resize the grid once the user
        // finishes scrubbing instead of rebuilding thousands of cells per move.
        if !editing { apply(Int(value)) }
      }
      .accessibilityLabel(axis == 1 ? LocalizedStringKey("Grid width") : "Grid height")
      .accessibilityValue("\(Int(value))")
      HStack {
        Text("2")
        Spacer()
        Text("256")
      }.font(.caption).foregroundStyle(.secondary)
    }.padding(20)
  }
}

private struct PatternFile: Identifiable {
  let url: URL
  let modified: Date
  var id: URL { url }
  var name: String { url.deletingPathExtension().lastPathComponent }
}

// A named pattern from the built-in Life Lexicon library (demo/gol/patterns/library.json).
private struct BuiltinPattern: Decodable, Identifiable {
  let name: String
  let category: String
  let width: Int
  let height: Int
  let period: Int?
  let speed: String?
  let rle: String
  var id: String { category + ":" + name }

  static let categories: [(id: String, title: String)] = [
    ("still", String(localized: "Still lifes")), ("oscillator", String(localized: "Oscillators")),
    ("spaceship", String(localized: "Spaceships")), ("gun", String(localized: "Guns")),
    ("puffer", String(localized: "Puffers and rakes")), ("methuselah", String(localized: "Methuselahs")),
    ("other", String(localized: "Other structures")),
  ]

  static let all: [BuiltinPattern] = {
    struct Document: Decodable { let patterns: [BuiltinPattern] }
    guard
      let url = Bundle.main.resourceURL?.appendingPathComponent(
        "SparkiumResources/Patterns/library.json"),
      let data = try? Data(contentsOf: url),
      let document = try? JSONDecoder().decode(Document.self, from: data)
    else { return [] }
    return document.patterns
  }()

  var detail: String {
    var parts: [String] = []
    if let speed {
      parts.append(
        speed.replacingOccurrences(of: " diagonal", with: " " + String(localized: "diagonal"))
          .replacingOccurrences(of: " orthogonal", with: " " + String(localized: "orthogonal"))
          .replacingOccurrences(of: " oblique", with: " " + String(localized: "oblique")))
    }
    if let period, period > 1 { parts.append("p\(period)") }
    parts.append("\(width)×\(height)")
    return parts.joined(separator: " · ")
  }

  // Life .cells text for the run-length encoded cells.
  var cells: String {
    var rows: [String] = []
    var row = ""
    var count = ""
    for c in rle {
      if c.isNumber {
        count.append(c)
        continue
      }
      let n = Int(count) ?? 1
      count = ""
      switch c {
      case "b": row += String(repeating: ".", count: n)
      case "o": row += String(repeating: "O", count: n)
      case "$", "!":
        rows.append(row)
        if c == "$" { rows.append(contentsOf: Array(repeating: "", count: n - 1)) }
        row = ""
      default: break
      }
      if c == "!" { break }
    }
    let lines = rows.map { $0.padding(toLength: width, withPad: ".", startingAt: 0) }
    return "!Name: \(name)\n!Life Lexicon, CC BY-SA 3.0\n" + lines.joined(separator: "\n") + "\n"
  }
}

// Life patterns saved inside the app, listed before the system document picker.
private enum PatternLibrary {
  static var directory: URL {
    let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Patterns", isDirectory: true)
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  static func files() -> [PatternFile] {
    let urls =
      (try? FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
    return urls.filter { $0.pathExtension.lowercased() == "cells" }.map { url in
      let modified =
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
        ?? .distantPast
      return PatternFile(url: url, modified: modified)
    }.sorted { $0.modified > $1.modified }
  }

  // A file in the library, without directories or reserved characters.
  static func url(named name: String) -> URL? {
    var base = name.trimmingCharacters(in: .whitespacesAndNewlines)
    if base.lowercased().hasSuffix(".cells") { base.removeLast(6) }
    base = base.components(separatedBy: CharacterSet(charactersIn: "/\\:*?\"<>|")).joined(separator: "_")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return base.isEmpty ? nil : directory.appendingPathComponent(base).appendingPathExtension("cells")
  }
}

private struct PatternLibraryView: View {
  let saving: Bool
  let choose: (URL) -> Void
  let system: () -> Void
  let cancel: () -> Void
  @State private var files = PatternLibrary.files()
  @State private var name = "Life " + Date().formatted(
    .verbatim("\(year: .defaultDigits)-\(month: .twoDigits)-\(day: .twoDigits) \(hour: .twoDigits(clock: .twentyFourHour, hourCycle: .zeroBased)).\(minute: .twoDigits).\(second: .twoDigits)",
      timeZone: .current, calendar: .current))
  @State private var replacing: URL?
  @State private var query = ""
  @State private var folderQuery = ""
  // Smoke runs can open a folder directly with LONGMARCH_SMOKE_FOLDER.
  @State private var path = ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_FOLDER"].map { [$0] } ?? []

  // The folder of the patterns the user saved; the others are built-in categories.
  private static let mine = "mine"

  private static func matches(_ name: String, _ query: String) -> Bool {
    query.isEmpty || name.localizedCaseInsensitiveContains(query)
  }

  var body: some View {
    NavigationStack(path: $path) {
      root
        .navigationTitle(saving ? LocalizedStringKey("Save Pattern") : "Open Pattern")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
          ToolbarItem(placement: .cancellationAction) { Button("Cancel", action: cancel) }
          if saving {
            ToolbarItem(placement: .confirmationAction) {
              Button("Save", action: save).disabled(PatternLibrary.url(named: name) == nil)
            }
          }
        }
        .navigationDestination(for: String.self) { folder($0) }
    }
    .preferredColorScheme(.dark)
    .alert(
      "Replace Pattern?", isPresented: Binding(get: { replacing != nil }, set: { if !$0 { replacing = nil } })
    ) {
      Button("Cancel", role: .cancel) { replacing = nil }
      Button("Replace", role: .destructive) {
        if let url = replacing { choose(url) }
        replacing = nil
      }
    } message: {
      Text("“\(replacing?.deletingPathExtension().lastPathComponent ?? "")” already exists. Do you want to replace it?")
    }
  }

  @ViewBuilder private var root: some View {
    if saving {
      List {
        Section {
          TextField("File name", text: $name).submitLabel(.done).onSubmit(save)
        }
        savedSection(files, header: String(localized: "My Patterns"))
        systemSection
      }
    } else {
      List {
        if query.isEmpty {
          Section {
            folderLink(Self.mine, title: String(localized: "My Patterns"), count: files.count, symbol: "folder.fill.badge.person.crop")
            ForEach(BuiltinPattern.categories, id: \.id) { category in
              folderLink(
                category.id, title: category.title,
                count: BuiltinPattern.all.filter { $0.category == category.id }.count, symbol: "folder.fill")
            }
          }
        } else {
          let saved = files.filter { Self.matches($0.name, query) }
          if !saved.isEmpty { savedSection(saved, header: String(localized: "My Patterns") + " · \(saved.count)") }
          ForEach(BuiltinPattern.categories, id: \.id) { category in
            let items = builtins(category.id, query)
            if !items.isEmpty {
              Section("\(category.title) · \(items.count)") { ForEach(items) { builtinRow($0) } }
            }
          }
        }
        systemSection
      }
      .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search patterns")
    }
  }

  // One folder of the open sheet: the saved patterns or a built-in category.
  private func folder(_ id: String) -> some View {
    let title = id == Self.mine ? String(localized: "My Patterns") : BuiltinPattern.categories.first { $0.id == id }?.title ?? id
    return List {
      if id == Self.mine {
        let saved = files.filter { Self.matches($0.name, folderQuery) }
        if saved.isEmpty && folderQuery.isEmpty {
          Text("No saved patterns yet").foregroundStyle(.secondary)
        }
        savedRows(saved)
      } else {
        ForEach(builtins(id, folderQuery)) { builtinRow($0) }
      }
    }
    .navigationTitle(title)
    .searchable(text: $folderQuery, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search patterns")
    .onAppear { folderQuery = "" }
  }

  private func builtins(_ category: String, _ query: String) -> [BuiltinPattern] {
    BuiltinPattern.all.filter { $0.category == category && Self.matches($0.name, query) }
  }

  private func folderLink(_ id: String, title: String, count: Int, symbol: String) -> some View {
    NavigationLink(value: id) {
      HStack(spacing: 12) {
        Image(systemName: symbol).foregroundStyle(.tint)
        Text(title).foregroundStyle(Color(uiColor: .label))
        Spacer()
        Text("\(count)").foregroundStyle(Color(uiColor: .secondaryLabel)).monospacedDigit()
      }
    }
  }

  private func savedSection(_ saved: [PatternFile], header: String) -> some View {
    Section(header) {
      if saved.isEmpty {
        Text("No saved patterns yet").foregroundStyle(.secondary)
      }
      savedRows(saved)
    }
  }

  private func savedRows(_ saved: [PatternFile]) -> some View {
    ForEach(saved) { file in
      Button {
        if saving { replacing = file.url } else { choose(file.url) }
      } label: {
        HStack(spacing: 12) {
          Image(systemName: "doc.text").foregroundStyle(.tint)
          VStack(alignment: .leading, spacing: 2) {
            Text(file.name).foregroundStyle(Color(uiColor: .label)).lineLimit(1)
            Text(file.modified, format: .dateTime.year().month().day().hour().minute())
              .font(.footnote).foregroundStyle(Color(uiColor: .secondaryLabel))
          }
        }
      }
    }
    .onDelete { offsets in
      for index in offsets { try? FileManager.default.removeItem(at: saved[index].url) }
      files = PatternLibrary.files()
    }
  }

  private func builtinRow(_ pattern: BuiltinPattern) -> some View {
    Button {
      openBuiltin(pattern)
    } label: {
      HStack(spacing: 12) {
        Image(systemName: "square.grid.3x3.fill").foregroundStyle(.tint)
        VStack(alignment: .leading, spacing: 2) {
          Text(pattern.name).foregroundStyle(Color(uiColor: .label)).lineLimit(1)
          Text(pattern.detail).font(.footnote).foregroundStyle(Color(uiColor: .secondaryLabel))
        }
      }
    }
  }

  private var systemSection: some View {
    Section {
      Button(saving ? LocalizedStringKey("Save to Files…") : "Open from Files…", action: system)
    } footer: {
      if !saving { Text("Built-in patterns come from the Life Lexicon (Stephen A. Silver et al., CC BY-SA 3.0).") }
    }
  }

  private func openBuiltin(_ pattern: BuiltinPattern) {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("Builtin.cells")
    do {
      try pattern.cells.write(to: url, atomically: true, encoding: .utf8)
      choose(url)
    } catch {
      cancel()
    }
  }

  private func save() {
    guard let url = PatternLibrary.url(named: name) else { return }
    if FileManager.default.fileExists(atPath: url.path) { replacing = url } else { choose(url) }
  }
}

private struct DesktopGameView: UIViewRepresentable {
  let life: Bool
  let active: Bool
  var status: (Double, String?) -> Void
  final class Coordinator: NSObject, UIDocumentPickerDelegate,
    UIPopoverPresentationControllerDelegate
  {
    weak var view: GameMetalView?
    var error: (String?) -> Void = { _ in }
    private var exporting = false
    private var temporary: URL?
    private weak var sizePopover: UIViewController?
    private weak var library: UIViewController?
    private var libraryHandled = false

    func requestSize(_ axis: Int, value: Int) {
      guard let view, let root = view.window?.rootViewController,
        root.presentedViewController == nil
      else { return }
      let content = GridDimensionPicker(
        axis: axis, value: Double(value),
        apply: { [weak view] value in
          view?.renderer.setGridDimension(axis, value: value)
        },
        close: { [weak self] in
          self?.sizePopover?.dismiss(animated: true)
        })
      let controller = UIHostingController(rootView: content)
      controller.modalPresentationStyle = .popover
      controller.preferredContentSize = CGSize(width: min(340, view.bounds.width - 24), height: 216)
      if let popover = controller.popoverPresentationController {
        popover.sourceView = view
        popover.sourceRect = CGRect(
          origin: view.lastPointerPosition, size: CGSize(width: 1, height: 1))
        popover.permittedArrowDirections = [.up, .down]
        popover.delegate = self
      }
      sizePopover = controller
      root.present(controller, animated: true)
    }

    func adaptivePresentationStyle(
      for controller: UIPresentationController,
      traitCollection: UITraitCollection
    ) -> UIModalPresentationStyle { controller.presentedViewController === library ? .automatic : .none }

    // Open and save start in the in-app pattern library; the system document
    // picker is one link away. Dismissing the library cancels the request.
    func request(_ action: Int) {
      guard let view, let root = view.window?.rootViewController, root.presentedViewController == nil
      else {
        cancel()
        return
      }
      libraryHandled = false
      let content = PatternLibraryView(
        saving: action == 2,
        choose: { [weak self] url in self?.finishLibrary { self?.complete(url) } },
        system: { [weak self] in self?.finishLibrary { self?.systemRequest(action) } },
        cancel: { [weak self] in self?.finishLibrary { self?.cancel() } })
      let controller = UIHostingController(rootView: content)
      if let sheet = controller.sheetPresentationController {
        sheet.detents = [.medium(), .large()]
        sheet.prefersGrabberVisible = true
      }
      controller.presentationController?.delegate = self
      library = controller
      root.present(controller, animated: true)
    }

    private func finishLibrary(_ next: @escaping () -> Void) {
      libraryHandled = true
      if let library { library.dismiss(animated: true, completion: next) } else { next() }
    }

    func presentationControllerDidDismiss(_ controller: UIPresentationController) {
      guard controller.presentedViewController === library, !libraryHandled else { return }
      libraryHandled = true
      cancel()
    }

    private func complete(_ url: URL) {
      view?.renderer.completeFile(url.path) { [weak self] message in self?.error(message) }
    }

    private func systemRequest(_ action: Int) {
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
    view.renderer.sizeRequest = { [weak coordinator = context.coordinator] axis, value in
      coordinator?.requestSize(axis, value: value)
    }
    let resources = Bundle.main.resourceURL!.appendingPathComponent("SparkiumResources")
    view.renderer.start(view: view, resources: resources, demo: life ? "gol" : "2048") {
      _, _, fps, _, _, _, _, error in status(fps, error)
    }
    if let file = ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_FILE"] {
      // Opens the pattern library the way the open/save buttons do, for screenshots.
      DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak coordinator = context.coordinator] in
        coordinator?.request(file == "save" ? 2 : 1)
      }
    }
        if let axis = ProcessInfo.processInfo.environment["LONGMARCH_SMOKE_SIZE_PICKER"] {
      DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak view] in
        view?.smokeSizeTap(axis == "height" ? 2 : 1)
      }
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
        .defersSystemGestures(on: .bottom)
      } else {
        ZStack {
          // Match the renderer's cream background through the system safe areas.
          Color(red: 250.0 / 255, green: 248.0 / 255, blue: 240.0 / 255)
            .ignoresSafeArea()
          VStack(spacing: 8) {
            HStack(spacing: 12) {
              Text("[Metal] 2048 FPS: \(fps, specifier: "%.1f")")
                .font(.caption2.monospaced())
                .foregroundStyle(Color(red: 0.47, green: 0.44, blue: 0.40))
              Button("Demos", action: onExit).font(.caption)
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(.black.opacity(0.05), in: Capsule())
            // The native container fills the remaining space for swipe input;
            // its Metal canvas keeps the compact, top-aligned board layout.
            game
          }
          .padding(.top, 8).padding(.bottom, 8)
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
