import { common } from '@kit.AbilityKit';
import { window, display } from '@kit.ArkUI';
import { fileIo as fs, picker } from '@kit.CoreFileKit';
import { sensor } from '@kit.SensorServiceKit';
import native from 'liblongmarch.so';

interface SceneInfo { id: string; name: string; width: number; height: number; }
interface DemoInfo { id: string; name: string; detail: string; symbol: Resource; reason?: string; }
interface DemoSection { title: string; demos: DemoInfo[]; }
interface PatternFile { name: string; modified: number; }
// A named pattern from the built-in Life Lexicon library (demo/gol/patterns/library.json).
interface BuiltinPattern {
  name: string; category: string; width: number; height: number; period?: number; speed?: string; rle: string;
}
interface PatternDocument { patterns: BuiltinPattern[]; }
// One row of the library list: a section header, a saved pattern or a built-in one.
interface LibraryRow {
  key: string; header?: string; folder?: string; title?: string; count?: number;
  saved?: PatternFile; builtin?: BuiltinPattern;
}

const LEXICON_LICENSE: string = 'https://creativecommons.org/licenses/by-sa/3.0/';
const CATEGORY_ORDER: string[] = ['still', 'oscillator', 'spaceship', 'gun', 'puffer', 'methuselah', 'other'];

class LibraryRows implements IDataSource {
  private rows: LibraryRow[] = [];
  private listeners: DataChangeListener[] = [];
  set(rows: LibraryRow[]): void {
    this.rows = rows;
    this.listeners.forEach((listener: DataChangeListener) => listener.onDataReloaded());
  }
  totalCount(): number { return this.rows.length; }
  getData(index: number): LibraryRow { return this.rows[index]; }
  registerDataChangeListener(listener: DataChangeListener): void { this.listeners.push(listener); }
  unregisterDataChangeListener(listener: DataChangeListener): void {
    this.listeners = this.listeners.filter((item: DataChangeListener) => item !== listener);
  }
}

// Category names are the string resources category_<id>; speed directions are direction_<name>.
function patternDetail(pattern: BuiltinPattern, text: (name: string) => string): string {
  const parts: string[] = [];
  if (pattern.speed) {
    parts.push(pattern.speed.replace(/ (diagonal|orthogonal|oblique)$/,
      (match: string, direction: string): string => ` ${text(`direction_${direction}`)}`));
  }
  if (pattern.period && pattern.period > 1) parts.push(`p${pattern.period}`);
  parts.push(`${pattern.width}×${pattern.height}`);
  return parts.join(' · ');
}

// Life .cells text for a run-length encoded pattern.
function patternCells(pattern: BuiltinPattern): string {
  const rows: string[] = [];
  let row = '';
  let count = '';
  for (const c of pattern.rle) {
    if (c >= '0' && c <= '9') { count += c; continue; }
    const n = count.length > 0 ? Number(count) : 1;
    count = '';
    if (c === 'b') row += '.'.repeat(n);
    else if (c === 'o') row += 'O'.repeat(n);
    else if (c === '$' || c === '!') {
      rows.push(row);
      for (let i = 1; i < n && c === '$'; i++) rows.push('');
      row = '';
      if (c === '!') break;
    }
  }
  const lines = rows.map((line: string) => line.padEnd(pattern.width, '.'));
  return `!Name: ${pattern.name}\n!Life Lexicon (Stephen A. Silver et al.), adapted under CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/\n${lines.join('\n')}\n`;
}
// A single object-literal @Builder argument is passed by reference, so the
// builder refreshes when the state it reads changes; plain arguments do not.
interface CaptionArgs { text: string; color: string; }
interface MenuArgs { label: string; value: string; items: MenuElement[]; }
interface ToggleArgs { label: string; isOn: boolean; target: string; }
interface ButtonArgs { label: string; enabled: boolean; action: string; }
interface HeaderArgs { title: string; closeColor: string; }
interface NativeCommand {
  type: string; demo?: string; scene?: string; value?: number | boolean;
  hdr?: boolean; exposure?: number; particles?: number; galaxies?: number; step?: number;
  scale?: number; reset?: boolean; zoom?: number; x?: number; y?: number; yaw?: number; pitch?: number;
  axis?: number; path?: string; angle?: number; kind?: number; request?: number;
}
class NativeStatus {
  ready: boolean = false;
  hdr: boolean = false;
  error: string = '';
  demo: string = '';
  scene: string = '';
  device: string = '';
  pipeline: string = '';
  resources: string = '';
  scenes: SceneInfo[] = [];
  fps: number = 0;
  frames: number = 0;
  seconds: number = 0;
  spp: number = 0;
  width: number = 0;
  height: number = 0;
  bounces: number = 0;
  gpuMs: number = 0;
  fileRequest: number = 0;
  fileRevision: number = 0;
  sizeAxis: number = 0;
  sizeValue: number = 64;
}

// iOS dark-mode system colors, so both apps share one visual language.
const BACKGROUND = '#000000';
const GROUPED_ROW = '#1C1C1E';
const ROW_PRESSED = '#2C2C2E';
// Grouped rows on a sheet, one level above the sheet background.
const SHEET = '#1C1C1E';
const SHEET_ROW = '#2C2C2E';
const SEPARATOR = '#38383A';
const LABEL = '#FFFFFF';
const SECONDARY = '#99EBEBF5';
const HEADER = '#8D8D93';
const TINT = '#0A84FF';
const FILL = '#3D767680';
const SWITCH_ON = '#30D158';
const DESTRUCTIVE = '#FF453A';
const PANEL = '#BF000000';
const PANEL_STROKE = '#26FFFFFF';
const CREAM = '#FAF8F0';
// A hidden gesture indicator reports no avoid area; keep its touch strip clear.
const HIDDEN_INDICATOR_VP = 16;
const UNAVAILABLE = 'Requires a full RT pipeline, which the shared mobile adapter does not provide yet; use Ray Query.';

const GAMES: DemoInfo[] = [
  { id: '2048', name: '2048', detail: 'Swipe to merge · Expectimax AI', symbol: $r('sys.symbol.square_grid_2x2') },
  { id: 'gol', name: 'Game of Life', detail: 'Editable cells · Boundary modes · Pattern files', symbol: $r('sys.symbol.grid') }
];
const RENDERING: DemoInfo[] = [
  { id: 'sparkium', name: 'Sparkium', detail: 'Path tracing · 9 bundled scenes', symbol: $r('sys.symbol.perspective') }
];
const COMPUTE: DemoInfo[] = [
  { id: 'nbody_cs', name: 'NBody CS', detail: 'GPU gravity simulation with interactive galaxies', symbol: $r('sys.symbol.wand_and_stars') }
];
const GRAPHICS: DemoInfo[] = [
  { id: 'graphics_hello_cube', name: 'Hello Cube', detail: 'Depth-tested rotating cube', symbol: $r('sys.symbol.perspective') },
  { id: 'graphics_hello_hdr', name: 'Hello HDR', detail: 'Linear HDR gradient and SDR reference white', symbol: $r('sys.symbol.sun_max_fill') },
  { id: 'graphics_hello_ray_query', name: 'Hello Ray Query', detail: 'Animated triangle and procedural sphere', symbol: $r('sys.symbol.rays') },
  { id: 'graphics_hello_external_shader', name: 'External Shader', detail: 'External RT pipeline shaders', symbol: $r('sys.symbol.doc_text'), reason: UNAVAILABLE },
  { id: 'graphics_hello_triangle', name: 'Hello Triangle', detail: 'Indexed geometry and vertex colors', symbol: $r('sys.symbol.triangleshape') },
  { id: 'graphics_hello_texture', name: 'Hello Texture', detail: 'Texture sampling and depth attachment', symbol: $r('sys.symbol.square_and_square') },
  { id: 'graphics_hello_blend', name: 'Hello Blend', detail: 'Overlapping triangles and alpha blending', symbol: $r('sys.symbol.square_stack_3d') },
  { id: 'graphics_hello_resize', name: 'Hello Resize', detail: 'Rotating cube and resizable render targets', symbol: $r('sys.symbol.arrow_up_left_and_arrow_down_right') },
  { id: 'graphics_hello_sdr_sample', name: 'Hello SDR Sample', detail: 'Pixel patterns and display response', symbol: $r('sys.symbol.sun_max') },
  { id: 'graphics_hello_ray_tracing', name: 'Hello Ray Tracing', detail: 'Native ray tracing pipeline', symbol: $r('sys.symbol.rays'), reason: UNAVAILABLE },
  { id: 'graphics_hello_rt_multi_shader_group', name: 'RT Multi Shader Group', detail: 'Procedural intersections and callable shaders', symbol: $r('sys.symbol.a_3d_square_fill'),
    reason: 'Requires RT shader groups and procedural intersection shaders, which the shared mobile adapter does not provide yet.' }
];
// Same section order as the iOS browser.
const SECTIONS: DemoSection[] = [
  { title: 'Games', demos: GAMES },
  { title: 'Rendering', demos: RENDERING },
  { title: 'Compute', demos: COMPUTE },
  { title: 'Graphics', demos: GRAPHICS.filter((demo: DemoInfo) => demo.reason === undefined) },
  { title: 'Unavailable on HarmonyOS', demos: GRAPHICS.filter((demo: DemoInfo) => demo.reason !== undefined) }
];
const DEMOS: DemoInfo[] = GAMES.concat(RENDERING, COMPUTE, GRAPHICS);

function grouped(value: number): string {
  return Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function pad(value: number): string {
  return value < 10 ? `0${value}` : `${value}`;
}

function patternDate(seconds: number): string {
  const d = new Date(seconds * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// A file name for the in-app pattern library, without directories or reserved characters.
function patternName(name: string): string {
  const base = name.trim().replace(/\.cells$/i, '').replace(/[\\/:*?"<>|]/g, '_').trim();
  return base.length > 0 ? base : '';
}

function rectEnd(start: number, size: number): number {
  return size > 0 ? start + size : 0;
}

@Entry
@Component
struct Index {
  @State private selected: string = '';
  @State private title: string = 'LongMarch Demos';
  @State private subtitle: string = '';
  @State private stats: NativeStatus = new NativeStatus();
  @State private controls: boolean = true;
  @State private scene: string = 'cornell_box';
  @State private limit: number = 32;
  @State private hdr: boolean = true;
  @State private exposure: number = 0;
  @State private paused: boolean = false;
  @State private particles: number = 4096;
  @State private galaxies: number = 10;
  @State private step: number = 0.03;
  @State private renderScale: number = 1;
  @State private sizeAxis: number = 0;
  @State private sizeValue: number = 64;
  @State private sizeVisible: boolean = false;
  @State private safeTop: number = 0;
  @State private safeBottom: number = 0;
  @State private safeLeft: number = 0;
  @State private safeRight: number = 0;
  @State private rootWidth: number = 360;
  @State private rootHeight: number = 780;
  @State private canvasHeight: number = 0;
  @State private anchorX: number = 0;
  @State private anchorY: number = 0;
  @State private scrolled: boolean = false;
  private timer: number = -1;
  private fileBusy: boolean = false;
  // In-app pattern library shown by the GoL open/save buttons.
  @State private libraryVisible: boolean = false;
  @State private librarySave: boolean = false;
  @State private patterns: PatternFile[] = [];
  @State private saveName: string = '';
  @State private librarySearch: string = '';
  // '' is the folder list; 'saved' or a category id is an opened folder.
  @State private libraryFolder: string = '';
  private libraryHandled: boolean = false;
  private libraryRows: LibraryRows = new LibraryRows();
  private builtins?: BuiltinPattern[];
  private request: number = 0;
  private areaWidth: number = 1;
  private areaHeight: number = 1;
  private lastX: number = 0;
  private lastY: number = 0;
  private startX: number = 0;
  private startY: number = 0;
  private distance: number = 0;
  private button: number = -1;
  private multi: boolean = false;
  private zoom: number = 1;
  private panX: number = 0;
  private panY: number = 0;
  private yaw: number = 0;
  private pitch: number = 0;
  private sensorActive: boolean = false;
  private entryAngle: number = 0;
  private shownError: string = '';
  private win?: window.Window;
  private listScroller: Scroller = new Scroller();
  private context(): common.UIAbilityContext { return this.getUIContext().getHostContext() as common.UIAbilityContext; }
  // A localized string resource, with %s/%d arguments.
  private text(name: string, ...args: (string | number)[]): string {
    return this.context().resourceManager.getStringByNameSync(name, ...args);
  }
  private send(c: NativeCommand): void { native.command(JSON.stringify(c)); }
  private isGame(): boolean { return this.selected === '2048' || this.selected === 'gol'; }

  aboutToAppear(): void {
    const startup = AppStorage.get<string>('longmarchStartupError') ?? '';
    if (startup.length > 0) this.alert('LongMarch', startup);
    this.trackSafeArea();
    this.timer = setInterval(() => {
      const value = JSON.parse(native.status()) as NativeStatus;
      if (value.scenes !== undefined) {
        this.stats = value;
        const smoke = AppStorage.get<string>('longmarchSmokeDemo') ?? '';
        if (value.ready && smoke.length > 0) {
          AppStorage.set('longmarchSmokeDemo', '');
          const demo = DEMOS.find((item: DemoInfo) => item.id === smoke);
          const scene = AppStorage.get<string>('longmarchSmokeScene') ?? '';
          if (value.scenes.some((item: SceneInfo) => item.id === scene)) this.scene = scene;
          if (demo && demo.reason === undefined) {
            this.open(demo);
            if (demo.id === 'sparkium') {
              const samples = AppStorage.get<number>('longmarchSmokeSamples') ?? 32;
              this.limit = Number.isFinite(samples) ? Math.max(1, Math.min(4096, Math.floor(samples))) : 32;
              this.send({ type: 'limit', value: this.limit });
            }
          }
        }
        if (value.sizeAxis !== this.sizeAxis) {
          this.sizeAxis = value.sizeAxis;
          this.sizeValue = value.sizeValue;
          this.sizeVisible = value.sizeAxis > 0;
        }
        if (this.selected === 'gol' && value.fileRequest > 0 && !this.fileBusy) this.fileAction(value.fileRequest);
        this.showNativeError(value.error);
      }
    }, 250);
  }
  aboutToDisappear(): void {
    if (this.timer >= 0) clearInterval(this.timer);
    if (this.win) {
      this.win.off('avoidAreaChange');
      this.win.off('windowSizeChange');
    }
    this.leave();
  }
  onBackPress(): boolean {
    if (this.selected.length > 0) { this.leave(); return true; }
    return false;
  }

  // Layout is fullscreen with hidden system bars, like the iOS app. Content that
  // iOS keeps inside its safe area avoids the camera cutout and gesture strip here.
  private async trackSafeArea(): Promise<void> {
    try {
      this.win = await window.getLastWindow(this.context());
      this.updateSafeArea();
      this.win.on('avoidAreaChange', () => this.updateSafeArea());
      this.win.on('windowSizeChange', () => this.updateSafeArea());
    } catch (e) { /* Without window metrics the layout keeps zero insets. */ }
  }
  private updateSafeArea(): void {
    if (!this.win) return;
    try {
      this.applySafeArea(this.win);
    } catch (e) { /* Keep the previous insets if the window is being torn down. */ }
  }
  private applySafeArea(win: window.Window): void {
    const ui = this.getUIContext();
    const rect = win.getWindowProperties().windowRect;
    const cutout = win.getWindowAvoidArea(window.AvoidAreaType.TYPE_CUTOUT);
    const system = win.getWindowAvoidArea(window.AvoidAreaType.TYPE_SYSTEM);
    const indicator = win.getWindowAvoidArea(window.AvoidAreaType.TYPE_NAVIGATION_INDICATOR);
    const top = Math.max(rectEnd(cutout.topRect.top, cutout.topRect.height), rectEnd(system.topRect.top, system.topRect.height));
    const left = rectEnd(cutout.leftRect.left, cutout.leftRect.width);
    const right = cutout.rightRect.width > 0 ? rect.width - cutout.rightRect.left : 0;
    const bottom = Math.max(indicator.bottomRect.height > 0 ? rect.height - indicator.bottomRect.top : 0,
      cutout.bottomRect.height > 0 ? rect.height - cutout.bottomRect.top : 0);
    this.safeTop = ui.px2vp(top);
    // iOS reserves the same horizontal inset on both landscape edges.
    this.safeLeft = this.safeRight = ui.px2vp(Math.max(left, right));
    this.safeBottom = ui.px2vp(bottom);
    this.sendBottomInset();
  }
  // GoL keeps its lower controls above the gesture area plus 12 vp, as on iOS.
  private sendBottomInset(): void {
    if (this.areaHeight <= 1) return;
    const gesture = this.safeBottom > 0 ? this.safeBottom : HIDDEN_INDICATOR_VP;
    this.send({ type: 'bottomInset', value: (gesture + 12) / this.areaHeight });
  }

  private alert(title: string, message: string): void {
    this.getUIContext().showAlertDialog({
      title: title, message: message, autoCancel: false,
      confirm: { value: $r('app.string.ok'), action: () => {} }
    });
  }
  // Games report errors in an alert; renderers show them in their control panel.
  private showNativeError(error: string): void {
    if (error.length === 0 || error === this.shownError) return;
    this.shownError = error;
    if (this.selected.length === 0 || this.isGame()) this.alert(this.isGame() ? this.text('game_error') : 'LongMarch', error);
    else this.controls = true;
  }

  private async orientation(id: string): Promise<void> {
    try {
      const win = await window.getLastWindow(this.context());
      let orientation = window.Orientation.AUTO_ROTATION;
      if (id === 'sparkium') orientation = window.Orientation.LANDSCAPE;
      if (id === 'gol') {
        const rotation = display.getDefaultDisplaySync().rotation;
        this.entryAngle = rotation * Math.PI / 2;
        if (rotation === 1) orientation = window.Orientation.LANDSCAPE;
        else if (rotation === 3) orientation = window.Orientation.LANDSCAPE_INVERTED;
        else if (rotation === 2) orientation = window.Orientation.PORTRAIT_INVERTED;
        else orientation = window.Orientation.PORTRAIT;
      }
      await win.setPreferredOrientation(orientation);
    } catch (e) { this.alert('LongMarch', `Orientation: ${e}`); }
  }
  private open(demo: DemoInfo): void {
    if (demo.reason !== undefined) return;
    if (!this.stats.ready) { this.alert('LongMarch', 'Bundled resources are still being prepared.'); return; }
    this.shownError = '';
    this.selected = demo.id; this.title = demo.name; this.subtitle = demo.detail;
    this.controls = true;
    this.paused = false; this.zoom = 1; this.panX = 0; this.panY = 0;
    this.button = -1; this.multi = false;
    this.send({ type: 'open', demo: demo.id, scene: this.scene });
    this.orientation(demo.id);
    if (demo.id === 'gol') this.startGravity();
  }
  private leave(): void {
    this.send({ type: 'stop' });
    this.selected = ''; this.sizeAxis = 0; this.sizeVisible = false; this.shownError = '';
    this.stopGravity();
    this.orientation('');
  }
  // GoL turns its direction icons with the device. The sensor is used only while
  // the page is visible: apps must not read sensors in the background.
  private startGravity(): void {
    if (this.sensorActive) return;
    try {
      sensor.on(sensor.SensorId.GRAVITY, (data: sensor.GravityResponse) => {
        if (Math.abs(data.x) + Math.abs(data.y) > 3) {
          const angle = Math.round(Math.atan2(-data.x, data.y) / (Math.PI / 2)) * Math.PI / 2;
          this.send({ type: 'icons', angle: this.entryAngle - angle });
        }
      }, { interval: 200000000 });
      this.sensorActive = true;
    } catch (e) { /* Direction icons remain upright if no gravity sensor is available. */ }
  }
  private stopGravity(): void {
    if (!this.sensorActive) return;
    try { sensor.off(sensor.SensorId.GRAVITY); } catch (e) { /* Already unsubscribed. */ }
    this.sensorActive = false;
  }
  onPageShow(): void {
    if (this.selected === 'gol') this.startGravity();
  }
  onPageHide(): void {
    this.stopGravity();
  }
  private selectScene(index: number): void {
    this.scene = this.stats.scenes[index].id;
    this.paused = false; this.shownError = '';
    this.zoom = 1; this.panX = 0; this.panY = 0;
    this.send({ type: 'open', demo: 'sparkium', scene: this.scene });
  }
  private reload(): void {
    this.zoom = 1; this.panX = 0; this.panY = 0; this.paused = false; this.shownError = '';
    this.send({ type: 'open', demo: 'sparkium', scene: this.scene });
  }
  private display(): void { this.send({ type: 'display', hdr: this.hdr, exposure: this.exposure }); }
  private configure(reset: boolean = false): void {
    this.send({ type: 'nbody', particles: this.particles, galaxies: this.galaxies,
      step: this.step, scale: this.renderScale, reset: reset });
  }
  private setSimulate(value: boolean): void {
    this.paused = !value;
    this.send({ type: 'pause', value: this.paused });
  }
  private applySize(value: number): void {
    this.sizeValue = Math.max(2, Math.min(256, Math.round(value)));
    this.send({ type: 'size', axis: this.sizeAxis, value: this.sizeValue });
  }
  private closeSize(): void {
    this.sizeVisible = false;
    this.send({ type: 'dismissSize' });
  }
  // 2048 draws into a top-aligned canvas; the rest of the page only accepts swipes.
  private canvasSpan(): number {
    return this.selected === '2048' && this.canvasHeight > 0 ? this.canvasHeight : this.areaHeight;
  }
  private pointer(kind: number, x: number, y: number, value: number = 0): void {
    this.send({ type: 'input', kind: kind, x: x / this.areaWidth, y: y / this.canvasSpan(), value: value });
  }
  private touch(event: TouchEvent): void {
    const points = event.touches;
    if (event.type === TouchType.Cancel) {
      if (this.button >= 0) { this.pointer(5, 0, 0, 0); this.pointer(5, 0, 0, 1); }
      this.button = -1; this.multi = false; this.distance = 0;
      return;
    }
    const ending = event.type === TouchType.Up;
    // Every change of the finger count keeps the pointer where the game last saw
    // it: a jump while a drag is active would move the grid by the difference.
    if (points.length === 0 || (ending && points.length <= 1)) {
      // A two-finger gesture ends at its last midpoint, a single touch where it lifted.
      if (!this.multi && event.changedTouches.length > 0) {
        this.lastX = event.changedTouches[0].x; this.lastY = event.changedTouches[0].y;
      }
      if (this.button >= 0) this.pointer(2, this.lastX, this.lastY, this.button);
      if (this.selected === '2048' && !this.multi) {
        const dx = this.lastX - this.startX, dy = this.lastY - this.startY;
        if (Math.max(Math.abs(dx), Math.abs(dy)) > 24) {
          this.pointer(4, 0, 0, Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 262 : 263) : (dy > 0 ? 264 : 265));
        }
      }
      this.button = -1; this.multi = false; this.distance = 0;
      return;
    }
    let x = points[0].x, y = points[0].y;
    let distance = 0;
    if (points.length >= 2) {
      x = (points[0].x + points[1].x) / 2;
      y = (points[0].y + points[1].y) / 2;
      distance = Math.hypot(points[1].x - points[0].x, points[1].y - points[0].y);
    }
    if (event.type === TouchType.Down && points.length === 1) {
      this.startX = this.lastX = x; this.startY = this.lastY = y;
      this.anchorX = x; this.anchorY = y;
      const onCanvas = this.selected !== '2048' || y <= this.canvasSpan();
      if (this.isGame() && onCanvas) { this.button = 0; this.pointer(1, x, y); }
    } else if (points.length >= 2 && !this.multi) {
      // Cancel the one-finger press where that finger was, then start the
      // two-finger pan at the midpoint.
      if (this.button >= 0) { this.pointer(5, this.lastX, this.lastY, 0); this.pointer(5, this.lastX, this.lastY, 1); }
      this.multi = true; this.button = -1;
      if (this.selected === 'gol') { this.button = 1; this.pointer(1, x, y, 1); }
    } else if (event.type === TouchType.Move && this.multi && points.length < 2) {
      // One finger of a two-finger gesture lifted: end it at the last midpoint.
      if (this.button >= 0) { this.pointer(2, this.lastX, this.lastY, this.button); this.button = -1; }
      return;
    } else if (event.type === TouchType.Move) {
      if (this.selected === 'sparkium') {
        const factor = this.distance > 0 && distance > 0 ? distance / this.distance : 1;
        const next = Math.min(16, Math.max(1, this.zoom * factor));
        const ratio = next / this.zoom;
        this.panX = (this.panX - (x - this.areaWidth / 2)) * ratio + x - this.areaWidth / 2 + x - this.lastX;
        this.panY = (this.panY - (y - this.areaHeight / 2)) * ratio + y - this.areaHeight / 2 + y - this.lastY;
        this.zoom = next;
        this.send({ type: 'view', zoom: this.zoom, x: this.getUIContext().vp2px(this.panX), y: this.getUIContext().vp2px(this.panY) });
      } else if (this.selected === 'nbody_cs') {
        // One degree per vp of drag, matching the iOS camera gesture.
        this.yaw += (x - this.lastX) * Math.PI / 180; this.pitch += (y - this.lastY) * Math.PI / 180;
        this.send({ type: 'rotate', yaw: this.yaw, pitch: this.pitch });
      } else if (this.button >= 0) {
        this.pointer(0, x, y);
        if (this.selected === 'gol' && this.distance > 0 && distance > 0) this.pointer(6, x, y, distance / this.distance);
      }
    }
    if (ending && this.button >= 0) {
      this.pointer(2, this.multi ? this.lastX : x, this.multi ? this.lastY : y, this.button);
      this.button = -1;
    }
    this.lastX = x; this.lastY = y; this.distance = distance;
  }
  private async completeFile(path: string): Promise<void> {
    this.request = Math.max(this.request, this.stats.fileRevision) + 1;
    const id = this.request;
    this.send({ type: 'file', path: path, request: id });
    for (let i = 0; i < 1200; i++) {
      await new Promise<void>((resolve) => setTimeout(resolve, 100));
      const value = JSON.parse(native.status()) as NativeStatus;
      if (value.fileRevision >= id) {
        if (value.error.length > 0) throw new Error(value.error);
        return;
      }
      if (this.selected !== 'gol') return;
    }
    throw new Error('File operation timed out');
  }
  private patternDirectory(): string {
    const directory = `${this.context().filesDir}/patterns`;
    if (!fs.accessSync(directory)) fs.mkdirSync(directory);
    return directory;
  }
  private refreshPatterns(): void {
    const directory = this.patternDirectory();
    this.patterns = fs.listFileSync(directory)
      .filter((file: string) => file.toLowerCase().endsWith('.cells'))
      .map((file: string): PatternFile => ({ name: file.slice(0, -6), modified: fs.statSync(`${directory}/${file}`).mtime }))
      .sort((a: PatternFile, b: PatternFile) => b.modified - a.modified);
  }
  private loadBuiltins(): BuiltinPattern[] {
    if (this.builtins === undefined) {
      try {
        const text = fs.readTextSync(`${this.stats.resources}/Patterns/library.json`);
        this.builtins = (JSON.parse(text) as PatternDocument).patterns;
      } catch (e) {
        this.builtins = [];
      }
    }
    return this.builtins;
  }
  // Saved patterns first, then the built-in library by category, filtered by the search text.
  private folderTitle(folder: string): string {
    return this.text(folder === 'saved' ? 'my_patterns' : `category_${folder}`);
  }
  // The library opens on a folder list: saved patterns in "My Patterns", then one
  // folder per built-in category. Searching the list spans every folder;
  // searching an opened folder filters only that folder.
  private rebuildRows(): void {
    const query = this.librarySearch.trim().toLowerCase();
    const matches = (name: string): boolean => query.length === 0 || name.toLowerCase().includes(query);
    const rows: LibraryRow[] = [];
    const folder = this.librarySave ? 'saved' : this.libraryFolder;
    const builtins = this.librarySave ? [] : this.loadBuiltins();
    if (folder === '' && query.length === 0) {
      rows.push({ key: 'f:saved', folder: 'saved', title: this.folderTitle('saved'), count: this.patterns.length });
      for (const category of CATEGORY_ORDER) {
        const count = builtins.filter((pattern: BuiltinPattern) => pattern.category === category).length;
        rows.push({ key: `f:${category}`, folder: category, title: this.folderTitle(category), count: count });
      }
      this.libraryRows.set(rows);
      return;
    }
    const grouped = folder === '';
    if (folder === '' || folder === 'saved') {
      const saved = this.patterns.filter((pattern: PatternFile) => matches(pattern.name));
      if (grouped && saved.length > 0) rows.push({ key: 'h:saved', header: `${this.folderTitle('saved')} · ${saved.length}` });
      saved.forEach((pattern: PatternFile) => rows.push({ key: `s:${pattern.name}`, saved: pattern }));
    }
    for (const category of CATEGORY_ORDER) {
      if (folder !== '' && folder !== category) continue;
      const items = builtins.filter((pattern: BuiltinPattern) => pattern.category === category && matches(pattern.name));
      if (items.length === 0) continue;
      if (grouped) rows.push({ key: `h:${category}`, header: `${this.folderTitle(category)} · ${items.length}` });
      items.forEach((pattern: BuiltinPattern) => rows.push({ key: `b:${category}:${pattern.name}`, builtin: pattern }));
    }
    this.libraryRows.set(rows);
  }
  private openFolder(folder: string): void {
    this.libraryFolder = folder;
    this.librarySearch = '';
    this.rebuildRows();
  }
  // Opening or saving starts in the app's own pattern library; the system
  // document picker is one link away.
  private fileAction(action: number): void {
    this.fileBusy = true;
    this.libraryHandled = false;
    this.librarySave = action !== 1;
    this.librarySearch = '';
    this.libraryFolder = '';
    try {
      this.refreshPatterns();
    } catch (e) {
      this.patterns = [];
    }
    this.rebuildRows();
    const now = new Date();
    this.saveName = `Life ${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ` +
      `${pad(now.getHours())}.${pad(now.getMinutes())}.${pad(now.getSeconds())}`;
    this.libraryVisible = true;
  }
  // Runs one choice from the library after closing it; cancelling reaches the
  // game as an empty path, exactly like a dismissed system picker.
  private async finishFile(work: () => Promise<void>): Promise<void> {
    this.libraryHandled = true;
    this.libraryVisible = false;
    try {
      await work();
    } catch (e) {
      this.alert(this.text('game_error'), `Files: ${e}`);
      try { await this.completeFile(''); } catch (cancelError) { /* Keep the original error visible. */ }
    }
    finally { this.fileBusy = false; }
  }
  // The built-in patterns' license (CC BY-SA 3.0), from the library footer.
  private openLicense(): void {
    this.context().openLink(LEXICON_LICENSE).catch(() => { /* The address stays readable in the footer. */ });
  }
  private libraryDismissed(): void {
    if (this.libraryHandled) return;
    this.libraryHandled = true;
    this.finishFile(() => this.completeFile(''));
  }
  private openPattern(pattern: PatternFile): void {
    this.finishFile(() => this.completeFile(`${this.patternDirectory()}/${pattern.name}.cells`));
  }
  private openBuiltin(pattern: BuiltinPattern): void {
    this.finishFile(async () => {
      const path = `${this.context().cacheDir}/Builtin.cells`;
      const file = fs.openSync(path, fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
      try { fs.writeSync(file.fd, patternCells(pattern)); } finally { fs.closeSync(file); }
      await this.completeFile(path);
    });
  }
  private savePattern(name: string): void {
    const base = patternName(name);
    if (base.length === 0) { this.alert(this.text('save_pattern'), this.text('enter_file_name')); return; }
    const save = () => this.finishFile(() => this.completeFile(`${this.patternDirectory()}/${base}.cells`));
    if (!this.patterns.some((pattern: PatternFile) => pattern.name === base)) { save(); return; }
    this.getUIContext().showAlertDialog({
      title: $r('app.string.replace_pattern'), message: this.text('replace_exists', base), autoCancel: true,
      primaryButton: { value: $r('app.string.cancel'), action: () => {} },
      secondaryButton: { value: $r('app.string.replace'), fontColor: DESTRUCTIVE, action: () => save() }
    });
  }
  private deletePattern(pattern: PatternFile): void {
    try {
      fs.unlinkSync(`${this.patternDirectory()}/${pattern.name}.cells`);
      this.refreshPatterns();
      this.rebuildRows();
    } catch (e) { this.alert(this.text('game_error'), `Files: ${e}`); }
  }
  private systemFile(): void {
    const save = this.librarySave;
    this.finishFile(async () => {
      const temp = `${this.context().cacheDir}/Life.cells`;
      const dialog = new picker.DocumentViewPicker(this.context());
      if (!save) {
        const options = new picker.DocumentSelectOptions(); options.maxSelectNumber = 1;
        const uris = await dialog.select(options);
        if (uris.length === 0) { await this.completeFile(''); return; }
        const file = fs.openSync(uris[0], fs.OpenMode.READ_ONLY);
        try { await fs.copyFile(file.fd, temp); } finally { fs.closeSync(file); }
        await this.completeFile(temp);
      } else {
        await this.completeFile(temp);
        const options = new picker.DocumentSaveOptions(); options.newFileNames = ['Life.cells'];
        const uris = await dialog.save(options);
        if (uris.length > 0) {
          const file = fs.openSync(uris[0], fs.OpenMode.READ_WRITE);
          try { await fs.copyFile(temp, file.fd); } finally { fs.closeSync(file); }
        }
      }
    });
  }

  private sceneItems(): MenuElement[] {
    return this.stats.scenes.map((s: SceneInfo, index: number): MenuElement => ({ value: s.name, action: () => this.selectScene(index) }));
  }
  private limitItems(): MenuElement[] {
    return [1, 2, 8, 32, 64, 128, 256, 512, 1024, 4096].map((n: number): MenuElement => ({
      value: n.toString(), action: () => { this.limit = n; this.send({ type: 'limit', value: n }); }
    }));
  }
  private particleItems(): MenuElement[] {
    return [1024, 2048, 4096, 8192, 16384, 32768, 65536].map((n: number): MenuElement => ({
      value: grouped(n), action: () => { this.particles = n; this.configure(); }
    }));
  }
  private scaleItems(): MenuElement[] {
    return [0.25, 0.5, 1].map((n: number): MenuElement => ({
      value: `${n * 100}%`, action: () => { this.renderScale = n; this.configure(); }
    }));
  }
  private sceneName(): string {
    return this.stats.scenes.find((s: SceneInfo) => s.id === this.scene)?.name ?? this.scene;
  }
  private sparkiumStatus(): string {
    if (this.stats.error.length > 0) return 'Render failed';
    if (this.stats.frames === 0) return 'Loading scene and preparing Vulkan…';
    if (this.stats.spp >= this.limit) return 'Sample limit reached';
    return 'Rendering';
  }
  private hdrCaption(): string {
    if (!this.hdr) return this.selected === 'sparkium' ? 'SDR · scene view transform' : 'SDR output';
    return this.stats.hdr ? 'HDR10 · BT.2020 / PQ · 203-nit reference white' : 'HDR requested · the display reports SDR output';
  }
  private panelWidth(): number { return Math.min(350, this.rootWidth - this.safeLeft - this.safeRight - 20); }
  private panelHeight(): number { return Math.max(0, this.rootHeight - this.safeTop - this.safeBottom - 74); }

  // ---- Browser: inline navigation bar and inset grouped list ----

  @Builder
  private demoRow(demo: DemoInfo, last: boolean) {
    Column() {
      Row({ space: 16 }) {
        Row() { SymbolGlyph(demo.symbol).fontSize(22).fontColor([TINT]) }.width(32).justifyContent(FlexAlign.Center)
        Column({ space: 4 }) {
          Text(demo.name).fontSize(17).fontWeight(600).fontColor(LABEL)
          Text(demo.reason ?? demo.detail).fontSize(15).fontColor(SECONDARY)
        }.alignItems(HorizontalAlign.Start).layoutWeight(1)
        SymbolGlyph(demo.reason === undefined ? $r('sys.symbol.chevron_right') : $r('sys.symbol.ohos_lock'))
          .fontSize(18).fontWeight(FontWeight.Medium).fontColor([SECONDARY])
      }.padding({ left: 16, right: 16, top: 17, bottom: 17 }).width('100%')
      .opacity(demo.reason === undefined ? 1 : 0.45)
      if (!last) { Divider().strokeWidth(0.5).color(SEPARATOR).margin({ left: 64, right: 16 }) }
    }
    .enabled(demo.reason === undefined)
    .stateStyles({ normal: { .backgroundColor(GROUPED_ROW) }, pressed: { .backgroundColor(ROW_PRESSED) },
      disabled: { .backgroundColor(GROUPED_ROW) } })
    .onClick(() => this.open(demo))
    .accessibilityText(demo.name)
  }

  @Builder
  private browser() {
    Stack({ alignContent: Alignment.Top }) {
      Scroll(this.listScroller) {
        Column() {
          if (!this.stats.ready) {
            Row({ space: 8 }) {
              LoadingProgress().color(SECONDARY).width(18).height(18)
              Text('Preparing bundled resources…').fontSize(13).fontColor(SECONDARY)
            }.margin({ bottom: 4 })
          }
          ForEach(SECTIONS, (section: DemoSection) => {
            Column() {
              Text(section.title).fontSize(17).fontWeight(600).fontColor(HEADER)
                .padding({ left: 16, bottom: 10 })
              Column() {
                ForEach(section.demos, (demo: DemoInfo, index: number) => {
                  this.demoRow(demo, index === section.demos.length - 1)
                }, (demo: DemoInfo) => demo.id)
              }.backgroundColor(GROUPED_ROW).borderRadius(26).clip(true)
            }.alignItems(HorizontalAlign.Start).width('100%').margin({ top: 26 })
          }, (section: DemoSection) => section.title)
        }
        .padding({ left: 16 + this.safeLeft, right: 16 + this.safeRight, top: this.safeTop + 44 - 7, bottom: this.safeBottom + 24 })
        .width('100%')
      }
      .scrollBar(BarState.Off).edgeEffect(EdgeEffect.Spring).width('100%').height('100%')
      .onDidScroll(() => { this.scrolled = this.listScroller.currentOffset().yOffset > 1; })
      // Inline title bar: transparent at the top edge, material once content scrolls under it.
      Column() {
        Text('LongMarch Demos').fontSize(17).fontWeight(600).fontColor(LABEL).height(44)
      }
      .padding({ top: this.safeTop }).width('100%').alignItems(HorizontalAlign.Center)
      .backgroundColor(this.scrolled ? '#B3000000' : '#00000000')
      .backgroundBlurStyle(this.scrolled ? BlurStyle.Thin : BlurStyle.NONE)
      .border({ width: { bottom: this.scrolled ? 0.5 : 0 }, color: SEPARATOR })
      .hitTestBehavior(HitTestMode.Transparent)
    }.width('100%').height('100%').backgroundColor(BACKGROUND)
  }

  // ---- Shared iOS-style controls ----

  @Builder
  private menuRow(args: MenuArgs) {
    Row() {
      Text(args.label).fontSize(17).fontColor(LABEL)
      Blank()
      Row({ space: 5 }) {
        Text(args.value).fontSize(17).fontColor(TINT).textAlign(TextAlign.End).maxLines(2)
        Column() {
          SymbolGlyph($r('sys.symbol.chevron_up')).fontSize(9).fontColor([TINT])
          SymbolGlyph($r('sys.symbol.chevron_down')).fontSize(9).fontColor([TINT]).margin({ top: -3 })
        }
      }.constraintSize({ maxWidth: '62%' }).bindMenu(args.items)
    }.width('100%').constraintSize({ minHeight: 38 })
  }

  @Builder
  private toggleRow(args: ToggleArgs) {
    Row() {
      Text(args.label).fontSize(17).fontColor(LABEL)
      Blank()
      Toggle({ type: ToggleType.Switch, isOn: args.isOn }).selectedColor(SWITCH_ON).switchPointColor('#FFFFFF')
        .onChange((value: boolean) => {
          if (args.target === 'hdr') { this.hdr = value; this.display(); } else this.setSimulate(value);
        })
    }.width('100%')
  }

  @Builder
  private stepper(target: string) {
    Row() {
      Row() { SymbolGlyph($r('sys.symbol.minus')).fontSize(17).fontColor([LABEL]) }
        .width(47).height(32).justifyContent(FlexAlign.Center).onClick(() => this.adjust(target, -1))
      Divider().vertical(true).height(18).strokeWidth(1).color('#5A5A5E')
      Row() { SymbolGlyph($r('sys.symbol.plus')).fontSize(17).fontColor([LABEL]) }
        .width(47).height(32).justifyContent(FlexAlign.Center).onClick(() => this.adjust(target, 1))
    }.backgroundColor(FILL).borderRadius(16)
  }
  private adjust(target: string, delta: number): void {
    if (target === 'galaxies') {
      this.galaxies = Math.max(1, Math.min(20, this.galaxies + delta));
      this.configure();
    } else this.applySize(this.sizeValue + delta);
  }

  @Builder
  private borderedButton(args: ButtonArgs) {
    Text(args.label).fontSize(17).fontColor(args.enabled ? TINT : '#4DEBEBF5')
      .padding({ left: 14, right: 14, top: 7, bottom: 7 }).backgroundColor(FILL).borderRadius(100)
      .onClick(() => {
        if (!args.enabled) return;
        if (args.action === 'reload') this.reload(); else this.send({ type: 'resetFilm' });
      })
  }

  @Builder
  private caption(args: CaptionArgs) {
    Text(args.text).fontSize(12).fontColor(args.color).fontFeature('"tnum" on')
  }

  @Builder
  private panelHeader(args: HeaderArgs) {
    Row() {
      Text(args.title).fontSize(17).fontWeight(600).fontColor(LABEL)
      Blank()
      Row() { SymbolGlyph($r('sys.symbol.xmark')).fontSize(20).fontColor([args.closeColor]) }
        .width(44).height(44).justifyContent(FlexAlign.Center).onClick(() => this.controls = false)
        .accessibilityText('Hide controls')
    }.width('100%').padding({ left: 14, right: 2 })
  }

  // ---- Graphics / NBody control panel (iOS GraphicsDemoView) ----

  @Builder
  private graphicsPanel() {
    Column() {
      this.panelHeader({ title: this.title, closeColor: TINT })
      Divider().strokeWidth(0.5).color(SEPARATOR)
      Scroll() {
        Column({ space: 12 }) {
          Text(this.subtitle).fontSize(15).fontColor(SECONDARY)
          if (this.selected === 'graphics_hello_hdr' || this.selected === 'nbody_cs') {
            this.toggleRow({ label: 'HDR display', isOn: this.hdr, target: 'hdr' })
          }
          if (this.selected === 'nbody_cs') {
            this.menuRow({ label: 'Particles', value: grouped(this.particles), items: this.particleItems() })
            Row() {
              Text(`Galaxies: ${this.galaxies}`).fontSize(17).fontColor(LABEL)
              Blank()
              this.stepper('galaxies')
            }.width('100%')
            Text(`Time step: ${this.step.toFixed(3)}`).fontSize(17).fontColor(LABEL)
            Slider({ value: this.step, min: .001, max: .1, step: .001 })
              .selectedColor(TINT).trackColor('#3A3A3C').blockColor('#FFFFFF').trackThickness(4)
              .onChange((value: number) => { this.step = value; this.configure(); })
            Text('Reset particles').fontSize(17).fontColor(TINT).onClick(() => this.configure(true))
            Text('Drag the image to rotate the camera.').fontSize(12).fontColor(SECONDARY)
          }
          if (this.selected === 'nbody_cs' || this.selected === 'graphics_hello_resize') {
            this.toggleRow({ label: this.selected === 'nbody_cs' ? 'Simulate' : 'Animate', isOn: !this.paused, target: 'simulate' })
            this.menuRow({ label: 'Render scale', value: `${this.renderScale * 100}%`, items: this.scaleItems() })
          }
          Divider().strokeWidth(0.5).color(SEPARATOR)
          Column({ space: 6 }) {
            this.caption({ text: this.stats.frames === 0 ? 'Preparing Vulkan…' : `Frame: ${grouped(this.stats.frames)}`, color: LABEL })
            this.caption({ text: `FPS: ${this.stats.fps.toFixed(1)} · Frame: ${this.stats.fps > 0 ? (1000 / this.stats.fps).toFixed(2) : '—'} ms`, color: LABEL })
            this.caption({ text: `GPU: ${this.stats.gpuMs > 0 ? this.stats.gpuMs.toFixed(2) + ' ms' : 'unavailable'}`, color: LABEL })
            if (this.selected === 'nbody_cs' && !this.paused) {
              this.caption({ text: `Estimated: ${(this.particles * this.particles * 20 * this.stats.fps / 1e9).toFixed(1)} GFLOP/s`, color: LABEL })
            }
            this.caption({ text: `Resolution: ${grouped(this.stats.width)} × ${grouped(this.stats.height)}`, color: LABEL })
            this.caption({ text: 'Backend: Vulkan', color: LABEL })
            this.caption({ text: `Device: ${this.stats.device || '—'}`, color: LABEL })
            this.caption({ text: `Pipeline: ${this.selected === 'nbody_cs' ? 'Compute + Rasterization' :
              this.selected === 'graphics_hello_ray_query' ? 'Compute + Ray Query' : 'Rasterization'}`, color: LABEL })
            if (this.selected === 'graphics_hello_hdr' || this.selected === 'nbody_cs') this.caption({ text: this.hdrCaption(), color: SECONDARY })
            if (this.stats.error.length > 0) {
              Text(this.stats.error).fontSize(12).fontColor(DESTRUCTIVE).copyOption(CopyOptions.LocalDevice)
            }
          }.alignItems(HorizontalAlign.Start).width('100%')
        }.alignItems(HorizontalAlign.Start).padding(14).width('100%')
      }.scrollBar(BarState.Off).edgeEffect(EdgeEffect.Spring).align(Alignment.Top).layoutWeight(1)
    }
    .width(this.panelWidth()).height(this.panelHeight())
    .backgroundColor(PANEL).backgroundBlurStyle(BlurStyle.Thin)
    .borderRadius(12).border({ width: 1, color: PANEL_STROKE }).clip(true)
  }

  // ---- Sparkium control panel (iOS RenderViewport) ----

  @Builder
  private sparkiumPanel() {
    Column() {
      this.panelHeader({ title: 'Sparkium scenes', closeColor: LABEL })
      Divider().strokeWidth(0.5).color(SEPARATOR)
      Scroll() {
        Column({ space: 10 }) {
          this.menuRow({ label: 'Scene', value: this.sceneName(), items: this.sceneItems() })
          this.menuRow({ label: 'SPP limit', value: this.limit.toString(), items: this.limitItems() })
          this.toggleRow({ label: 'HDR display', isOn: this.hdr, target: 'hdr' })
          Text(`Exposure: ${this.exposure.toFixed(1)} EV`).fontSize(12).fontColor(LABEL)
          Slider({ value: this.exposure, min: -5, max: 5, step: .1 })
            .selectedColor(TINT).trackColor('#3A3A3C').blockColor('#FFFFFF').trackThickness(4)
            .onChange((value: number) => { this.exposure = value; this.display(); })
          Text(this.hdrCaption()).fontSize(12).fontColor(SECONDARY)
          Row({ space: 8 }) {
            this.borderedButton({ label: 'Reload', enabled: true, action: 'reload' })
            this.borderedButton({ label: 'Reset film', enabled: this.stats.frames > 0 && this.stats.error.length === 0, action: 'reset' })
          }
          Divider().strokeWidth(0.5).color(SEPARATOR)
          Column({ space: 6 }) {
            Row() {
              this.caption({ text: `Ray/s: ${(this.stats.width * this.stats.height * this.stats.fps / 1e6).toFixed(2)} M`, color: LABEL })
              Blank()
              this.caption({ text: `FPS: ${this.stats.fps.toFixed(1)}`, color: LABEL })
            }.width('100%')
            this.caption({ text: `Accumulated spp: ${this.stats.spp}`, color: LABEL })
            Progress({ value: Math.min(this.stats.spp, this.limit), total: this.limit, type: ProgressType.Linear })
              .color(TINT).backgroundColor('#3A3A3C').style({ strokeWidth: 4 }).width('100%')
            this.caption({ text: this.sparkiumStatus(), color: SECONDARY })
            this.caption({ text: `Render time: ${this.stats.seconds.toFixed(1)} s`, color: LABEL })
            this.caption({ text: 'Backend: Vulkan', color: LABEL })
            this.caption({ text: `Device: ${this.stats.device || '—'}`, color: LABEL })
            this.caption({ text: `Pipeline: ${this.stats.pipeline || '—'}`, color: LABEL })
            this.caption({ text: this.stats.width > 0 ? `Resolution: ${grouped(this.stats.width)} × ${grouped(this.stats.height)}` : 'Resolution: loading…', color: LABEL })
            this.caption({ text: `Samples / frame: 1 · Max bounces: ${this.stats.bounces}`, color: LABEL })
            this.caption({ text: 'Ray/s and FPS describe the last rendered frame.', color: SECONDARY })
            this.caption({ text: `scenes/${this.scene}/scene.json`, color: SECONDARY })
            if (this.stats.error.length > 0) {
              Text(this.stats.error).fontSize(12).fontColor(DESTRUCTIVE).copyOption(CopyOptions.LocalDevice)
            }
          }.alignItems(HorizontalAlign.Start).width('100%')
        }.alignItems(HorizontalAlign.Start).padding(14).width('100%')
      }.scrollBar(BarState.Off).edgeEffect(EdgeEffect.Spring).align(Alignment.Top).layoutWeight(1)
      Divider().strokeWidth(0.5).color(SEPARATOR)
      Text('Pinch to zoom · Drag to pan · Double-tap to fit').fontSize(11).fontColor(SECONDARY).padding(12)
    }
    .width(this.panelWidth()).height(this.panelHeight())
    .backgroundColor(PANEL).backgroundBlurStyle(BlurStyle.Thin)
    .borderRadius(12).border({ width: 1, color: PANEL_STROKE }).clip(true)
    .shadow({ radius: 16, color: '#66000000', offsetY: 6 })
  }

  // ---- Renderer page: full-screen surface, sidebar panel and Demos button ----

  @Builder
  private surface() {
    XComponent({ id: 'longmarchSurface', type: XComponentType.SURFACE, libraryname: 'longmarch' })
      .width('100%').height(this.selected === '2048' && this.canvasHeight > 0 ? this.canvasHeight : '100%')
    // Native-library XComponents dispatch touch to C++; a sibling ArkUI
    // input layer keeps all multi-touch handling in the shared UI adapter.
    Column() {}.width('100%').height('100%').backgroundColor('#00000000')
      .hitTestBehavior(HitTestMode.Block)
      .onTouch((event: TouchEvent) => this.touch(event))
      .parallelGesture(TapGesture({ count: 2 }).onAction(() => {
        if (this.selected === 'sparkium') {
          this.zoom = 1; this.panX = 0; this.panY = 0; this.send({ type: 'view', zoom: 1, x: 0, y: 0 });
        }
      }))
    // Anchor for the GoL size popover at the tapped readout, like the iOS popover source.
    Column().width(1).height(1).position({ x: this.anchorX, y: this.anchorY }).hitTestBehavior(HitTestMode.None)
      .bindPopup(this.sizeVisible, {
        builder: this.sizePicker(),
        placement: Placement.Top,
        enableArrow: true,
        autoCancel: true,
        popupColor: '#F22C2C2E',
        backgroundBlurStyle: BlurStyle.NONE,
        radius: 26,
        width: Math.min(340, this.rootWidth - 24),
        onStateChange: (e: PopupStateChangeParam) => { if (!e.isVisible && this.sizeVisible) this.closeSize(); }
      })
  }

  @Builder
  private sizePicker() {
    Column({ space: 16 }) {
      Row() {
        Text(this.sizeAxis === 1 ? $r('app.string.grid_width') : $r('app.string.grid_height')).fontSize(17).fontWeight(600).fontColor(LABEL)
        Blank()
        Text($r('app.string.done')).fontSize(17).fontColor(TINT).onClick(() => { this.applySize(this.sizeValue); this.closeSize(); })
      }.width('100%')
      Row() {
        Text(`${this.sizeValue}`).fontSize(22).fontColor(LABEL).fontFeature('"tnum" on')
        Blank()
        this.stepper('size')
      }.width('100%')
      // Resize the grid once scrubbing ends instead of rebuilding cells on every move.
      Slider({ value: this.sizeValue, min: 2, max: 256, step: 1 })
        .selectedColor(TINT).trackColor('#48484A').blockColor('#FFFFFF').trackThickness(4)
        .onChange((value: number, mode: SliderChangeMode) => {
          this.sizeValue = Math.round(value);
          if (mode === SliderChangeMode.End || mode === SliderChangeMode.Click) this.applySize(value);
        })
        .accessibilityText(this.text(this.sizeAxis === 1 ? 'grid_width' : 'grid_height'))
      Row() {
        Text('2').fontSize(12).fontColor(SECONDARY)
        Blank()
        Text('256').fontSize(12).fontColor(SECONDARY)
      }.width('100%')
    }.padding(20).width('100%')
  }

  @Builder
  private demosButton() {
    Row({ space: 6 }) {
      SymbolGlyph($r('sys.symbol.list_bullet')).fontSize(17).fontColor([LABEL])
      Text('Demos').fontSize(17).fontColor(LABEL)
    }
    .padding({ left: 14, right: 14 }).height(34).backgroundColor(TINT).borderRadius(17)
    .onClick(() => this.leave()).accessibilityText('Demos')
  }

  @Builder
  private rendererOverlay() {
    if (this.controls) {
      Column() {
        if (this.selected === 'sparkium') { this.sparkiumPanel() } else { this.graphicsPanel() }
      }.position({ x: this.safeLeft + 10, y: this.safeTop + 64 })
    } else {
      Row() {
        SymbolGlyph($r('sys.symbol.open_sidebar')).fontSize(20)
          .fontColor([this.selected === 'sparkium' ? LABEL : TINT])
      }
      .width(44).height(44).justifyContent(FlexAlign.Center)
      .backgroundColor(this.selected === 'sparkium' ? '#99202022' : FILL)
      .backgroundBlurStyle(this.selected === 'sparkium' ? BlurStyle.Thin : BlurStyle.NONE)
      .borderRadius(this.selected === 'sparkium' ? 10 : 22)
      .position({ x: this.safeLeft + 10, y: this.safeTop + 10 })
      .onClick(() => this.controls = true).accessibilityText('Show controls')
    }
    Row() { this.demosButton() }
      .position({ x: this.rootWidth - this.safeRight - 12, y: this.safeTop + 12 }).markAnchor({ x: '100%', y: 0 })
  }

  // ---- GoL pattern library (open/save start here, not in the system picker) ----

  @Builder
  private patternDelete(pattern: PatternFile) {
    Row() { Text($r('app.string.delete')).fontSize(17).fontColor(LABEL) }
    .width(80).height('100%').justifyContent(FlexAlign.Center).backgroundColor(DESTRUCTIVE)
    .onClick(() => this.deletePattern(pattern))
  }

  @Builder
  private libraryRow(row: LibraryRow) {
    if (row.folder !== undefined) {
      Row({ space: 12 }) {
        SymbolGlyph(row.folder === 'saved' ? $r('sys.symbol.folder_fill') : $r('sys.symbol.folder'))
          .fontSize(22).fontColor([TINT])
        Text(row.title).fontSize(17).fontColor(LABEL).layoutWeight(1)
        Text(`${row.count}`).fontSize(17).fontColor(SECONDARY)
        SymbolGlyph($r('sys.symbol.chevron_right')).fontSize(16).fontColor([SECONDARY])
      }
      .width('100%').padding({ left: 16, right: 16, top: 12, bottom: 12 }).backgroundColor(SHEET_ROW)
      .border({ width: { bottom: 0.5 }, color: SEPARATOR })
    } else if (row.header !== undefined) {
      Text(row.header).fontSize(15).fontWeight(600).fontColor(HEADER)
        .width('100%').padding({ left: 16, right: 16, top: 18, bottom: 8 })
    } else {
      Row({ space: 12 }) {
        SymbolGlyph(row.saved ? $r('sys.symbol.doc_text') : $r('sys.symbol.square_grid_2x2'))
          .fontSize(20).fontColor([TINT])
        Column({ space: 2 }) {
          Text(row.saved ? row.saved.name : row.builtin!.name).fontSize(17).fontColor(LABEL)
            .maxLines(1).textOverflow({ overflow: TextOverflow.Ellipsis })
          Text(row.saved ? patternDate(row.saved.modified) : patternDetail(row.builtin!, (name: string): string => this.text(name)))
            .fontSize(13).fontColor(SECONDARY)
        }.alignItems(HorizontalAlign.Start).layoutWeight(1)
      }
      .width('100%').padding({ left: 16, right: 16, top: 10, bottom: 10 }).backgroundColor(SHEET_ROW)
      .border({ width: { bottom: 0.5 }, color: SEPARATOR })
    }
  }

  @Builder
  private patternLibrary() {
    Column() {
      if (this.librarySave) {
        Row({ space: 12 }) {
          TextInput({ text: this.saveName, placeholder: $r('app.string.file_name') })
            .layoutWeight(1).height(40).fontSize(17).fontColor(LABEL).placeholderColor(SECONDARY)
            .backgroundColor(FILL).borderRadius(10).caretColor(TINT)
            .onChange((value: string) => { this.saveName = value; })
            .onSubmit(() => this.savePattern(this.saveName))
          Text($r('app.string.save')).fontSize(17).fontWeight(600).fontColor(TINT).onClick(() => this.savePattern(this.saveName))
        }.width('100%').padding({ left: 16, right: 16, bottom: 4 })
      } else {
        if (this.libraryFolder.length > 0) {
          Row({ space: 4 }) {
            SymbolGlyph($r('sys.symbol.chevron_left')).fontSize(18).fontColor([TINT])
            Text($r('app.string.all_folders')).fontSize(17).fontColor(TINT)
            Blank()
            Text(this.folderTitle(this.libraryFolder)).fontSize(17).fontWeight(600).fontColor(LABEL)
          }
          .width('100%').padding({ left: 16, right: 16, bottom: 10 })
          .onClick(() => this.openFolder(''))
        }
        Search({ value: this.librarySearch,
          placeholder: this.libraryFolder.length > 0 ?
            this.text('search_in_folder', this.folderTitle(this.libraryFolder)) : this.text('search_all') })
          .height(40).margin({ left: 16, right: 16 }).backgroundColor(FILL).fontColor(LABEL)
          .placeholderColor(SECONDARY).textFont({ size: 17 })
          .onChange((value: string) => { this.librarySearch = value; this.rebuildRows(); })
      }
      List() {
        LazyForEach(this.libraryRows, (row: LibraryRow) => {
          ListItem() { this.libraryRow(row) }
          .swipeAction(row.saved ? { end: this.patternDelete(row.saved) } : {})
          .onClick(() => {
            if (row.folder !== undefined) this.openFolder(row.folder);
            else if (row.saved) { if (this.librarySave) this.savePattern(row.saved.name); else this.openPattern(row.saved); }
            else if (row.builtin) this.openBuiltin(row.builtin);
          })
        }, (row: LibraryRow) => row.key)
        ListItem() {
          Text(this.librarySave ? '' : this.text('lexicon_credit'))
            .fontSize(12).fontColor(SECONDARY).padding({ left: 16, right: 16, top: 12, bottom: 12 })
            .onClick(() => { if (!this.librarySave) this.openLicense(); })
        }
      }
      .cachedCount(8).scrollBar(BarState.Auto).edgeEffect(EdgeEffect.Spring).layoutWeight(1).width('100%')
      Text(this.librarySave ? $r('app.string.save_to_files') : $r('app.string.open_from_files'))
        .fontSize(17).fontColor(TINT).padding({ top: 14, bottom: 14 + this.safeBottom })
        .onClick(() => this.systemFile())
    }.width('100%').height('100%')
  }

  // ---- Game pages (iOS GamesView) ----

  @Builder
  private gameStatus(background: string, color: string) {
    Row({ space: 12 }) {
      Text(`[Vulkan] ${this.selected === 'gol' ? 'GoL' : '2048'} FPS: ${this.stats.fps.toFixed(1)}`)
        .fontSize(11).fontColor(color).fontFamily('monospace').fontFeature('"tnum" on')
      Text('Demos').fontSize(12).fontColor(TINT).onClick(() => this.leave()).accessibilityText('Demos')
    }.padding({ left: 12, right: 12, top: 6, bottom: 6 }).backgroundColor(background).borderRadius(100)
  }

  @Builder
  private gamePage() {
    if (this.selected === 'gol') {
      // The canvas fills the display; only the small status overlay uses the safe area.
      Stack() { this.surface() }
        .width('100%').height('100%')
        .onAreaChange((oldArea: Area, area: Area) => this.canvasArea(area))
      Row() { this.gameStatus('#73000000', LABEL) }
        .position({ x: this.rootWidth / 2, y: this.safeTop + 4 }).markAnchor({ x: '50%', y: 0 })
    } else {
      // Cream fills the system areas; the board stays top-aligned below the status.
      Column({ space: 8 }) {
        this.gameStatus('#0D000000', '#786F66')
        Stack({ alignContent: Alignment.Top }) { this.surface() }
          .width('100%').layoutWeight(1)
          .onAreaChange((oldArea: Area, area: Area) => this.canvasArea(area))
      }
      .padding({ top: this.safeTop + 8, bottom: this.safeBottom + 8, left: this.safeLeft, right: this.safeRight })
      .width('100%').height('100%').backgroundColor(CREAM)
    }
  }
  private canvasArea(area: Area): void {
    this.areaWidth = Number(area.width); this.areaHeight = Number(area.height);
    this.canvasHeight = Math.min(this.areaHeight, this.areaWidth * (118.5 / 88));
    this.sendBottomInset();
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      if (this.selected.length === 0) {
        this.browser()
      } else if (this.isGame()) {
        this.gamePage()
      } else {
        Stack() { this.surface() }
          .width('100%').height('100%')
          .onAreaChange((oldArea: Area, area: Area) => {
            this.areaWidth = Number(area.width); this.areaHeight = Number(area.height);
          })
        this.rendererOverlay()
      }
    }
    .width('100%').height('100%').backgroundColor(BACKGROUND)
    .onAreaChange((oldArea: Area, area: Area) => {
      this.rootWidth = Number(area.width); this.rootHeight = Number(area.height);
    })
    .bindSheet($$this.libraryVisible, this.patternLibrary(), {
      detents: [SheetSize.MEDIUM, SheetSize.LARGE], dragBar: true, backgroundColor: SHEET,
      title: { title: this.librarySave ? $r('app.string.save_pattern') : $r('app.string.open_pattern') },
      onDisappear: () => this.libraryDismissed()
    })
  }
}
