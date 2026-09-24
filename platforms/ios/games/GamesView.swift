import SwiftUI
import UIKit
import UniformTypeIdentifiers

@MainActor
private final class GamesModel: ObservableObject {
  let controller = GameController()
  @Published var state: [AnyHashable: Any] = [:]
  @Published var error: String?
  @Published var fit = 0
  init() { refresh() }
  private var scoreTapStart = 0.0
  private var scoreTapCount = 0
  func scoreTapped() {
    if flag("ai") {
      action("ai")
      scoreTapCount = 0
      return
    }
    let now = CACurrentMediaTime()
    if scoreTapCount == 0 || now - scoreTapStart > 1 {
      scoreTapStart = now
      scoreTapCount = 0
    }
    scoreTapCount += 1
    if scoreTapCount == 5 {
      action("ai")
      scoreTapCount = 0
    }
  }
  func refresh() { state = controller.snapshot }
  func int(_ key: String) -> Int { (state[key] as? NSNumber)?.intValue ?? 0 }
  func flag(_ key: String) -> Bool { (state[key] as? NSNumber)?.boolValue ?? false }
  func action(_ action: String, _ value: Int = 0) {
    controller.action(action, value: value)
    refresh()
  }
}

private struct FrameDriver: UIViewRepresentable {
  var active: Bool
  var tick: (Double) -> Void
  final class Coordinator: NSObject {
    var tick: (Double) -> Void = { _ in }
    var link: CADisplayLink?
    var previous = 0.0
    @objc func frame(_ link: CADisplayLink) {
      tick(previous == 0 ? 0 : min(0.1, link.timestamp - previous))
      previous = link.timestamp
    }
  }
  func makeCoordinator() -> Coordinator { Coordinator() }
  func makeUIView(context: Context) -> UIView {
    let link = CADisplayLink(
      target: context.coordinator, selector: #selector(Coordinator.frame(_:)))
    context.coordinator.link = link
    link.add(to: .main, forMode: .common)
    return UIView()
  }
  func updateUIView(_ view: UIView, context: Context) {
    context.coordinator.tick = tick
    context.coordinator.link?.isPaused = !active
    if !active { context.coordinator.previous = 0 }
  }
  static func dismantleUIView(_ view: UIView, coordinator: Coordinator) {
    coordinator.link?.invalidate()
  }
}

private final class CellCanvas: UIView {
  var cells = Data()
  var columns = 32
  override func draw(_ rect: CGRect) {
    guard let ctx = UIGraphicsGetCurrentContext() else { return }
    ctx.setFillColor(UIColor(white: 0.07, alpha: 1).cgColor)
    ctx.fill(bounds)
    ctx.setFillColor(UIColor(white: 0.15, alpha: 1).cgColor)
    for i in cells.indices {
      ctx.fill(CGRect(x: (i % columns) * 16, y: (i / columns) * 16, width: 15, height: 15))
    }
    ctx.setFillColor(UIColor(white: 0.86, alpha: 1).cgColor)
    for (i, cell) in cells.enumerated() where cell != 0 {
      ctx.fill(CGRect(x: (i % columns) * 16, y: (i / columns) * 16, width: 15, height: 15))
    }
  }
}
private final class GridScroll: UIScrollView, UIScrollViewDelegate {
  let canvas = CellCanvas()
  var toggle: (Int) -> Void = { _ in }
  var revision = -1
  var lastPainted = -1
  var oldSize = CGSize.zero
  override init(frame: CGRect) {
    super.init(frame: frame)
    delegate = self
    addSubview(canvas)
    backgroundColor = .black
    contentInsetAdjustmentBehavior = .never
    panGestureRecognizer.minimumNumberOfTouches = 2
    let tap = UITapGestureRecognizer(target: self, action: #selector(tapped(_:)))
    canvas.addGestureRecognizer(tap)
    let double = UITapGestureRecognizer(target: self, action: #selector(fitGrid))
    double.numberOfTapsRequired = 2
    addGestureRecognizer(double)
    tap.require(toFail: double)
    let paint = UIPanGestureRecognizer(target: self, action: #selector(painted(_:)))
    paint.maximumNumberOfTouches = 1
    canvas.addGestureRecognizer(paint)
    canvas.isUserInteractionEnabled = true
  }
  required init?(coder: NSCoder) { fatalError() }
  @objc func tapped(_ gesture: UITapGestureRecognizer) {
    let p = gesture.location(in: canvas)
    let x = Int(p.x / 16)
    let y = Int(p.y / 16)
    if x >= 0 && x < canvas.columns && y >= 0 && y * canvas.columns + x < canvas.cells.count {
      toggle(y * canvas.columns + x)
    }
  }
  @objc func painted(_ gesture: UIPanGestureRecognizer) {
    if gesture.state == .began { lastPainted = -1 }
    guard gesture.state == .began || gesture.state == .changed else { return }
    let p = gesture.location(in: canvas)
    let x = Int(p.x / 16)
    let y = Int(p.y / 16)
    let index = y * canvas.columns + x
    if p.x >= 0 && p.y >= 0 && x < canvas.columns && index < canvas.cells.count
      && index != lastPainted
    {
      toggle(index)
      lastPainted = index
    }
  }
  @objc func fitGrid() {
    guard canvas.bounds.width > 0 && bounds.width > 0 else { return }
    minimumZoomScale =
      min(bounds.width / canvas.bounds.width, bounds.height / canvas.bounds.height) * 0.95
    maximumZoomScale = max(8, minimumZoomScale * 16)
    zoomScale = minimumZoomScale
    centerGrid()
    contentOffset = CGPoint(x: -contentInset.left, y: -contentInset.top)
  }
  func centerGrid() {
    contentInset = UIEdgeInsets(
      top: max(0, (bounds.height - canvas.frame.height) / 2),
      left: max(0, (bounds.width - canvas.frame.width) / 2), bottom: 0, right: 0)
  }
  override func layoutSubviews() {
    super.layoutSubviews()
    if oldSize != bounds.size {
      oldSize = bounds.size
      fitGrid()
    }
  }
  func viewForZooming(in scrollView: UIScrollView) -> UIView? { canvas }
  func scrollViewDidZoom(_ scrollView: UIScrollView) { centerGrid() }
}
private struct LifeGrid: UIViewRepresentable {
  @ObservedObject var model: GamesModel
  func makeUIView(context: Context) -> GridScroll { GridScroll() }
  func updateUIView(_ view: GridScroll, context: Context) {
    let size = CGSize(width: model.int("width") * 16, height: model.int("height") * 16)
    let reset = view.canvas.bounds.size != size || view.revision != model.fit
    if reset {
      view.zoomScale = 1
      view.canvas.frame = CGRect(origin: .zero, size: size)
      view.contentSize = size
      view.revision = model.fit
    }
    view.canvas.columns = model.int("width")
    view.canvas.cells = model.state["cells"] as? Data ?? Data()
    view.canvas.setNeedsDisplay()
    view.toggle = {
      model.controller.toggleCell($0)
      model.refresh()
    }
    if reset { view.fitGrid() }
  }
}
private struct RaisedButton: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.padding(12).frame(minWidth: 44, minHeight: 44)
      .background(
        LinearGradient(
          colors: [Color(white: configuration.isPressed ? 0.18 : 0.32), Color(white: 0.17)],
          startPoint: .topLeading, endPoint: .bottomTrailing),
        in: RoundedRectangle(cornerRadius: 12)
      )
      .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.12)))
      .shadow(color: .black.opacity(0.5), radius: configuration.isPressed ? 1 : 4, y: 3)
      .scaleEffect(configuration.isPressed ? 0.94 : 1).animation(
        .spring(response: 0.22), value: configuration.isPressed)
  }
}
private struct PatternDocument: FileDocument {
  static var readableContentTypes: [UTType] { [.plainText] }
  var text: String
  init(_ text: String) { self.text = text }
  init(configuration: ReadConfiguration) throws {
    text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
  }
  func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    FileWrapper(regularFileWithContents: Data(text.utf8))
  }
}

struct GamesView: View {
  let life: Bool
  let onExit: () -> Void
  @StateObject private var model = GamesModel()
  @Environment(\.scenePhase) private var phase
  @State private var importing = false
  @State private var exporting = false
  @State private var document = PatternDocument("")
  @AppStorage("2048-best") private var best = 0
  var body: some View {
    VStack(spacing: 12) {
      HStack {
        Text(life ? "Game of Life" : "2048").font(.title2.bold())
        Spacer()
        if life { Text("Gen \(model.int("generation"))").monospacedDigit() }
        Button("Demos", action: onExit).buttonStyle(.bordered)
      }
      if life {
        LifeGrid(model: model).clipShape(RoundedRectangle(cornerRadius: 12))
        lifeControls
        Text("Tap to edit · Pinch to zoom · Two fingers to pan · Double-tap to fit").font(.caption)
          .foregroundStyle(.secondary)
      } else {
        puzzle
      }
    }.padding().background(Color(white: 0.10))
      .background(
        FrameDriver(active: phase == .active) { dt in
          model.controller.tick(dt, life: life)
          model.refresh()
        }.frame(width: 0, height: 0)
      )
      .fileImporter(isPresented: $importing, allowedContentTypes: [.item]) { result in
        do {
          let url = try result.get()
          let access = url.startAccessingSecurityScopedResource()
          defer { if access { url.stopAccessingSecurityScopedResource() } }
          let data = try Data(contentsOf: url, options: .mappedIfSafe)
          guard data.count <= 1024 * 1024 else { throw CocoaError(.fileReadTooLarge) }
          model.error = model.controller.loadPattern(String(decoding: data, as: UTF8.self))
          model.fit += 1
          model.refresh()
        } catch { model.error = error.localizedDescription }
      }
      .fileExporter(
        isPresented: $exporting, document: document, contentType: .plainText,
        defaultFilename: "Life.cells"
      ) { result in
        if case .failure(let error) = result { model.error = error.localizedDescription }
      }
      .alert(
        "Cannot load or save",
        isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })
      ) {
        Button("OK") { model.error = nil }
      } message: {
        Text(model.error ?? "")
      }
      .onChange(of: model.int("score")) { _, score in best = max(best, score) }
  }
  private func action(_ name: String, _ symbol: String, color: Color = .white) -> some View {
    Button {
      model.action(name)
    } label: {
      Image(systemName: symbol).font(.title3).foregroundStyle(color).contentTransition(
        .symbolEffect(.replace))
    }.buttonStyle(RaisedButton()).accessibilityLabel(name)
  }
  private var lifeControls: some View {
    ScrollView(.horizontal, showsIndicators: false) {
      HStack(spacing: 12) {
        action("clear", "arrow.counterclockwise", color: .red)
        action("play", model.flag("playing") ? "pause.fill" : "play.fill", color: .green)
        action("random", "die.face.5.fill", color: .blue)
        ForEach(["width", "height"], id: \.self) { dimension in
          GeometryReader { geometry in
            let fraction = Double(model.int(dimension) - 2) / 198
            ZStack {
              RoundedRectangle(cornerRadius: 12).fill(
                LinearGradient(
                  colors: [.black.opacity(0.6), .gray.opacity(0.18)], startPoint: .top,
                  endPoint: .bottom))
              HStack(spacing: 0) {
                Rectangle().fill(
                  LinearGradient(
                    colors: [.gray.opacity(0.7), .gray.opacity(0.35)], startPoint: .top,
                    endPoint: .bottom)
                ).frame(width: geometry.size.width * fraction)
                Spacer(minLength: 0)
              }.clipShape(RoundedRectangle(cornerRadius: 12))
              Text("\(dimension == "width" ? "W" : "H") \(model.int(dimension))").font(
                .system(.caption, design: .monospaced)
              ).foregroundStyle(.white.opacity(0.45))
            }.contentShape(Rectangle()).gesture(
              DragGesture(minimumDistance: 0).onChanged { value in
                let n = min(200, max(2, Int((value.location.x / geometry.size.width) * 198 + 2)))
                model.controller.resizeWidth(
                  dimension == "width" ? n : model.int("width"),
                  height: dimension == "height" ? n : model.int("height"))
                model.refresh()
              })
          }.frame(width: 130, height: 48).accessibilityLabel(dimension)
        }
        Button {
          model.action("speed")
        } label: {
          Text(["1×", "2×", "5×", "ϟ"][model.int("speed")]).font(.title3.monospaced())
        }.buttonStyle(RaisedButton())
        Button {
          model.action("boundary")
        } label: {
          boundaryIcon
        }.buttonStyle(RaisedButton()).accessibilityLabel(
          model.flag("periodic") ? "Periodic boundary" : "Fixed boundary")
        Menu {
          Button("Import .cells file") { importing = true }
          ForEach(["295P5H1V1", "gosper-glider-gun"], id: \.self) { name in
            Button(name) {
              do {
                let url = Bundle.main.resourceURL!.appendingPathComponent(
                  "SparkiumResources/Patterns/" + name + ".cells")
                model.error = model.controller.loadPattern(
                  try String(contentsOf: url, encoding: .utf8))
                model.fit += 1
                model.refresh()
              } catch { model.error = error.localizedDescription }
            }
          }
        } label: {
          Image(systemName: "folder")
        }.buttonStyle(RaisedButton())
        Button {
          document = PatternDocument(model.controller.savePattern())
          exporting = true
        } label: {
          Image(systemName: "square.and.arrow.up")
        }.buttonStyle(RaisedButton())
      }.padding(5)
    }.fixedSize(horizontal: false, vertical: true)
  }
  private var boundaryIcon: some View {
    Canvas { context, size in
      let color = Color(white: 0.85)
      let cells = model.state["glider"] as? Data ?? Data()
      for i in cells.indices where cells[i] != 0 {
        context.fill(
          Path(CGRect(x: 6 + (i % 4) * 4, y: 6 + (i / 4) * 4, width: 4, height: 4)),
          with: .color(color))
      }
      for side in 0..<4 {
        var c = context
        c.translateBy(x: 14, y: 14)
        c.rotate(by: .degrees(Double(side) * 90))
        if model.flag("periodic") {
          c.fill(Path(CGRect(x: -14, y: -14, width: 8, height: 3)), with: .color(color))
          c.fill(Path(CGRect(x: 6, y: -14, width: 8, height: 3)), with: .color(color))
        } else {
          c.fill(Path(CGRect(x: -14, y: -14, width: 28, height: 3)), with: .color(color))
        }
      }
    }.frame(width: 28, height: 28)
  }
  private var puzzle: some View {
    VStack(spacing: 16) {
      HStack {
        Button {
          model.scoreTapped()
        } label: {
          Text("SCORE \(model.int("score"))").foregroundStyle(model.flag("ai") ? .red : .white)
        }
        Text("BEST \(best)")
        Spacer()
        Button("New game") { model.action("new") }.buttonStyle(RaisedButton())
        Toggle("AI", isOn: Binding(get: { model.flag("ai") }, set: { _ in model.action("ai") }))
          .fixedSize()
      }.font(.system(.subheadline, design: .rounded).bold())
      GeometryReader { geometry in
        let side = min(geometry.size.width, geometry.size.height)
        let tiles = model.state["tiles"] as? [Int] ?? Array(repeating: 0, count: 16)
        ZStack {
          LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 4), spacing: 8
          ) {
            ForEach(0..<16) { i in
              let value = tiles[i]
              RoundedRectangle(cornerRadius: 9).fill(tileColor(value)).aspectRatio(
                1, contentMode: .fit
              )
              .overlay(
                Text(value == 0 ? "" : "\(value)").font(
                  .system(size: side / 13, weight: .bold, design: .rounded)
                ).minimumScaleFactor(0.35).foregroundStyle(value <= 4 ? Color(white: 0.4) : .white)
                  .padding(3)
              )
              .animation(.easeOut(duration: 0.12), value: value)
            }
          }.padding(8).background(
            Color(red: 0.73, green: 0.68, blue: 0.63), in: RoundedRectangle(cornerRadius: 14))
          if model.flag("won") || model.flag("over") {
            VStack {
              Text(model.flag("won") ? "You win!" : "Game over").font(.largeTitle.bold())
              Button(model.flag("won") ? "Keep going" : "Try again") {
                model.action(model.flag("won") ? "continue" : "new")
              }.buttonStyle(.borderedProminent)
            }.frame(maxWidth: .infinity, maxHeight: .infinity).background(
              .ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
          }
        }.frame(width: side, height: side).position(
          x: geometry.size.width / 2, y: geometry.size.height / 2
        )
        .gesture(
          DragGesture(minimumDistance: 20).onEnded { value in
            let t = value.translation
            model.action(
              "move", abs(t.width) > abs(t.height) ? (t.width > 0 ? 3 : 2) : (t.height > 0 ? 1 : 0))
          })
      }
      Text("Swipe to move · Merge equal tiles · Five taps on the score toggle AI").font(.caption)
        .foregroundStyle(.secondary)
    }
  }
  private func tileColor(_ value: Int) -> Color {
    if value == 0 { return Color(white: 0.65).opacity(0.5) }
    if value <= 4 { return Color(red: 0.93, green: 0.89, blue: 0.82) }
    return Color(
      hue: max(0.06, 0.12 - log2(Double(value)) * 0.003),
      saturation: min(0.85, 0.45 + log2(Double(value)) * 0.035), brightness: 0.95)
  }
}
