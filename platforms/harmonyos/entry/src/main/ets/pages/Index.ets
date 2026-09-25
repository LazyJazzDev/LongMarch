import { common } from '@kit.AbilityKit';
import { window, display } from '@kit.ArkUI';
import { fileIo as fs, picker } from '@kit.CoreFileKit';
import { sensor } from '@kit.SensorServiceKit';
import native from 'liblongmarch.so';

interface SceneInfo { id: string; name: string; width: number; height: number; }
interface DemoInfo { id: string; name: string; detail: string; reason?: string; }
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
const DEMOS: DemoInfo[] = [
  { id: '2048', name: '2048', detail: 'Swipe to move · tap the score five times for AI' },
  { id: 'gol', name: 'Game of Life', detail: 'Draw cells · two fingers to pan and zoom' },
  { id: 'sparkium', name: 'Sparkium', detail: 'Nine scenes · progressive path tracing' },
  { id: 'graphics_hello_triangle', name: 'Triangle', detail: 'Vertex colors' },
  { id: 'graphics_hello_texture', name: 'Texture', detail: 'Texture sampling' },
  { id: 'graphics_hello_blend', name: 'Blend', detail: 'Alpha blending' },
  { id: 'graphics_hello_cube', name: 'Cube', detail: 'Animated depth-tested cube' },
  { id: 'graphics_hello_resize', name: 'Resize', detail: 'Drawable size and render scale' },
  { id: 'graphics_hello_sdr_sample', name: 'SDR Sample', detail: 'Reference white' },
  { id: 'graphics_hello_hdr', name: 'HDR', detail: 'Linear gradient and HDR/SDR display' },
  { id: 'graphics_hello_ray_query', name: 'Ray Query', detail: 'Triangle and procedural sphere · requires hardware support' },
  { id: 'nbody_cs', name: 'NBody CS', detail: 'GPU particle simulation · drag to rotate' },
  { id: 'graphics_hello_ray_tracing', name: 'Ray Tracing', detail: 'Full RT pipeline', reason: 'This module has no shared mobile adapter yet, matching the iOS browser limitation.' },
  { id: 'graphics_hello_external_shader', name: 'External Shader', detail: 'Full RT pipeline', reason: 'This module has no shared mobile adapter yet, matching the iOS browser limitation.' },
  { id: 'graphics_hello_rt_multi_shader_group', name: 'RT Multi Shader Group', detail: 'Full RT pipeline', reason: 'This module has no shared mobile adapter yet, matching the iOS browser limitation.' }
];

@Entry
@Component
struct Index {
  @State private selected: string = '';
  @State private title: string = 'LongMarch Demos';
  @State private stats: NativeStatus = new NativeStatus();
  @State private localError: string = '';
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
  private timer: number = -1;
  private fileBusy: boolean = false;
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
  private context(): common.UIAbilityContext { return this.getUIContext().getHostContext() as common.UIAbilityContext; }
  private send(c: NativeCommand): void { native.command(JSON.stringify(c)); }

  aboutToAppear(): void {
    this.localError = AppStorage.get<string>('longmarchStartupError') ?? '';
    this.timer = setInterval(() => {
      const value = JSON.parse(native.status()) as NativeStatus;
      if (value.scenes !== undefined) {
        this.stats = value;
        const smoke = AppStorage.get<string>('longmarchSmokeDemo') ?? '';
        if (value.ready && smoke.length > 0) {
          AppStorage.set('longmarchSmokeDemo', '');
          const demo = DEMOS.find((item: DemoInfo) => item.id === smoke);
          if (demo) this.open(demo);
        }
        if (value.sizeAxis !== this.sizeAxis) {
          this.sizeAxis = value.sizeAxis;
          this.sizeValue = value.sizeValue;
        }
        if (this.selected === 'gol' && value.fileRequest > 0 && !this.fileBusy) this.fileAction(value.fileRequest);
      }
    }, 250);
  }
  aboutToDisappear(): void {
    if (this.timer >= 0) clearInterval(this.timer);
    this.leave();
  }
  onBackPress(): boolean {
    if (this.selected.length > 0) { this.leave(); return true; }
    return false;
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
    } catch (e) { this.localError = `Orientation: ${e}`; }
  }
  private open(demo: DemoInfo): void {
    if (demo.reason) { this.localError = demo.reason; return; }
    this.localError = '';
    this.selected = demo.id; this.title = demo.name;
    this.controls = demo.id !== '2048' && demo.id !== 'gol';
    this.paused = false; this.zoom = 1; this.panX = 0; this.panY = 0;
    this.button = -1; this.multi = false;
    this.send({ type: 'open', demo: demo.id, scene: this.scene });
    this.orientation(demo.id);
    if (demo.id === 'gol') {
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
  }
  private leave(): void {
    this.send({ type: 'stop' });
    this.selected = ''; this.sizeAxis = 0; this.localError = '';
    if (this.sensorActive) { sensor.off(sensor.SensorId.GRAVITY); this.sensorActive = false; }
    this.orientation('');
  }
  private selectScene(index: number): void {
    this.scene = this.stats.scenes[index].id;
    this.paused = false;
    this.zoom = 1; this.panX = 0; this.panY = 0;
    this.send({ type: 'open', demo: 'sparkium', scene: this.scene });
  }
  private reload(): void {
    this.zoom = 1; this.panX = 0; this.panY = 0; this.paused = false;
    this.send({ type: 'open', demo: 'sparkium', scene: this.scene });
  }
  private display(): void { this.send({ type: 'display', hdr: this.hdr, exposure: this.exposure }); }
  private configure(reset: boolean = false): void {
    this.send({ type: 'nbody', particles: this.particles, galaxies: this.galaxies,
      step: this.step, scale: this.renderScale, reset: reset });
  }
  private pointer(kind: number, x: number, y: number, value: number = 0): void {
    this.send({ type: 'input', kind: kind, x: x / this.areaWidth, y: y / this.areaHeight, value: value });
  }
  private touch(event: TouchEvent): void {
    const points = event.touches;
    const ending = event.type === TouchType.Up || event.type === TouchType.Cancel;
    if (points.length === 0 || event.type === TouchType.Cancel || (ending && points.length <= 1)) {
      if (event.changedTouches.length > 0) {
        this.lastX = event.changedTouches[0].x; this.lastY = event.changedTouches[0].y;
      }
      if (this.button >= 0) this.pointer(2, this.lastX, this.lastY, this.button);
      if (this.selected === '2048' && !this.multi && event.type !== TouchType.Cancel) {
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
      if (this.selected === '2048' || this.selected === 'gol') { this.button = 0; this.pointer(1, x, y); }
    } else if (points.length >= 2 && !this.multi) {
      if (this.button >= 0) { this.pointer(5, x, y, 0); this.pointer(5, x, y, 1); }
      this.multi = true; this.button = -1;
      if (this.selected === 'gol') { this.button = 1; this.pointer(1, x, y, 1); }
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
        this.yaw += (x - this.lastX) * .01; this.pitch += (y - this.lastY) * .01;
        this.send({ type: 'rotate', yaw: this.yaw, pitch: this.pitch });
      } else if (this.button >= 0) {
        this.pointer(0, x, y);
        if (this.selected === 'gol' && this.distance > 0 && distance > 0) this.pointer(6, x, y, distance / this.distance);
      }
    }
    if (ending && this.button >= 0) { this.pointer(2, x, y, this.button); this.button = -1; }
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
  private async fileAction(action: number): Promise<void> {
    this.fileBusy = true;
    const temp = `${this.context().cacheDir}/Life.cells`;
    try {
      const dialog = new picker.DocumentViewPicker(this.context());
      if (action === 1) {
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
    } catch (e) {
      this.localError = `Files: ${e}`;
      try { await this.completeFile(''); } catch (cancelError) { /* Keep the original error visible. */ }
    }
    finally { this.fileBusy = false; }
  }

  @Builder
  private browser() {
    Column({ space: 12 }) {
      Text('LongMarch Demos').fontSize(28).fontWeight(FontWeight.Bold).fontColor('#FFFFFF')
      Text(this.stats.ready ? 'Games · Graphics · Simulation · Sparkium' : 'Preparing bundled resources…')
        .fontSize(14).fontColor('#B8C3D5')
      List({ space: 10 }) {
        ForEach(DEMOS, (demo: DemoInfo) => {
          ListItem() {
            Column({ space: 6 }) {
              Text(demo.name).fontSize(20).fontColor('#FFFFFF')
              Text(demo.detail).fontSize(13).fontColor('#B8C3D5')
            }.alignItems(HorizontalAlign.Start).width('100%').padding(18).backgroundColor('#242E40').borderRadius(14)
          }.onClick(() => { if (this.stats.ready) this.open(demo); })
        }, (demo: DemoInfo) => demo.id)
      }.layoutWeight(1)
    }.padding(20).width('100%').height('100%').alignItems(HorizontalAlign.Start)
  }

  @Builder
  private renderControls() {
    Scroll() {
      Column({ space: 12 }) {
        Row() {
          Text(this.title).fontSize(20).fontWeight(FontWeight.Bold).layoutWeight(1)
          Button('Hide').onClick(() => this.controls = false)
        }.width('100%')
        if (this.selected === 'sparkium') {
          Text('Scene').width('100%').textAlign(TextAlign.Start)
          Select(this.stats.scenes.map((s: SceneInfo): SelectOption => ({ value: s.name })))
            .value(this.stats.scenes.find((s: SceneInfo) => s.id === this.scene)?.name ?? this.scene)
            .onSelect((index: number) => this.selectScene(index))
          Text(`SPP limit: ${this.limit}`)
          Select([1, 2, 8, 32, 64, 128, 256, 512, 1024, 4096].map((n: number): SelectOption => ({ value: n.toString() })))
            .value(this.limit.toString()).onSelect((index: number, value: string) => {
              this.limit = Number(value); this.send({ type: 'limit', value: this.limit });
            })
          Text(`Exposure: ${this.exposure.toFixed(1)} EV`)
          Slider({ value: this.exposure, min: -5, max: 5, step: .1 })
            .onChange((value: number) => { this.exposure = value; this.display(); })
          Row({ space: 8 }) {
            Button('Reload').onClick(() => this.reload())
            Button('Reset film').onClick(() => this.send({ type: 'resetFilm' }))
          }
          Text(`Accumulated spp: ${this.stats.spp} / ${this.limit}`)
          Progress({ value: Math.min(this.stats.spp, this.limit), total: this.limit })
          Text(`Ray/s: ${(this.stats.width * this.stats.height * this.stats.fps / 1e6).toFixed(2)} M`)
          Text(`${this.stats.pipeline} · ${this.stats.bounces} bounces · 1 sample/frame`)
          Text('Pinch to zoom · drag to pan · double-tap to fit').fontSize(12)
        }
        if (this.selected === 'nbody_cs') {
          Text(`Estimated: ${(this.particles * this.particles * 20 * this.stats.fps / 1e9).toFixed(1)} GFLOP/s`)
          Text(`Particles: ${this.particles}`)
          Select([1024, 2048, 4096, 8192, 16384, 32768, 65536].map((n: number): SelectOption => ({ value: n.toString() })))
            .value(this.particles.toString()).onSelect((index: number, value: string) => { this.particles = Number(value); this.configure(); })
          Text(`Galaxies: ${this.galaxies}`)
          Slider({ value: this.galaxies, min: 1, max: 20, step: 1 }).onChange((value: number, mode: SliderChangeMode) => {
            this.galaxies = value; if (mode === SliderChangeMode.End || mode === SliderChangeMode.Click) this.configure();
          })
          Text(`Time step: ${this.step.toFixed(3)}`)
          Slider({ value: this.step, min: .001, max: .1, step: .001 }).onChange((value: number) => { this.step = value; this.configure(); })
          Button('Reset particles').onClick(() => this.configure(true))
        }
        if (this.selected === 'nbody_cs' || this.selected === 'graphics_hello_resize') {
          Text(`Render scale: ${(this.renderScale * 100).toFixed(0)}%`)
          Slider({ value: this.renderScale, min: .25, max: 1, step: .25 }).onChange((value: number) => { this.renderScale = value; this.configure(); })
        }
        Row({ space: 12 }) {
          Text('HDR display')
          Toggle({ type: ToggleType.Switch, isOn: this.hdr }).onChange((value: boolean) => { this.hdr = value; this.display(); })
        }
        Text(this.stats.hdr ? 'Linear HDR surface' : 'SDR display · HDR requires a supported surface').fontSize(12)
        if (this.selected === 'sparkium' || this.selected === 'nbody_cs' || this.selected === 'graphics_hello_resize') {
          Button(this.paused ? 'Resume' : 'Pause').onClick(() => { this.paused = !this.paused; this.send({ type: 'pause', value: this.paused }); })
        }
        Text(`Frame: ${this.stats.frames}`)
        Text(`FPS: ${this.stats.fps.toFixed(1)} · frame: ${this.stats.fps > 0 ? (1000 / this.stats.fps).toFixed(2) : '—'} ms`)
        Text(`GPU: ${this.stats.gpuMs > 0 ? this.stats.gpuMs.toFixed(2) + ' ms' : 'unavailable'}`)
        Text(`${this.stats.width} × ${this.stats.height} · ${this.stats.seconds.toFixed(1)} s`)
        Text(`Vulkan · ${this.stats.device}`).fontSize(12)
      }.padding(16).width('100%')
    }.width(320).constraintSize({ maxHeight: '85%' }).backgroundColor('#E6202939').borderRadius(14)
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      if (this.selected.length === 0) {
        this.browser()
      } else {
        Stack() {
          XComponent({ id: 'longmarchSurface', type: XComponentType.SURFACE, libraryname: 'longmarch' })
            .width('100%').height('100%')
          // Native-library XComponents dispatch touch to C++; a sibling ArkUI
          // input layer keeps all multi-touch handling in the shared UI adapter.
          Column() {}.width('100%').height('100%').backgroundColor('#00000000')
            .hitTestBehavior(HitTestMode.Block)
            .onTouch((event: TouchEvent) => this.touch(event))
            .gesture(TapGesture({ count: 2 }).onAction(() => {
              if (this.selected === 'sparkium') {
                this.zoom = 1; this.panX = 0; this.panY = 0; this.send({ type: 'view', zoom: 1, x: 0, y: 0 });
              }
            }))
        }.width('100%').height(this.selected === 'gol' || this.selected === '2048' ? 'calc(100% - 62vp)' : '100%')
          .margin({ top: this.selected === 'gol' || this.selected === '2048' ? 62 : 0 })
          .onAreaChange((oldArea: Area, area: Area) => { this.areaWidth = Number(area.width); this.areaHeight = Number(area.height); })
        Row({ space: 12 }) {
          Button('Demos').onClick(() => this.leave())
          Text(`${this.title} · ${this.stats.fps.toFixed(1)} FPS`).fontColor('#FFFFFF').fontSize(12)
          if (!this.controls && this.selected !== 'gol' && this.selected !== '2048') {
            Button('Controls').onClick(() => this.controls = true)
          }
        }.padding(8).backgroundColor('#B0202939').borderRadius(24).margin(8)
        if (this.controls) { Column() { this.renderControls() }.margin({ top: 62, left: 8 }) }
        if (this.sizeAxis > 0) {
          Column({ space: 16 }) {
            Text(`${this.sizeAxis === 1 ? 'Width' : 'Height'}: ${this.sizeValue}`).fontSize(22)
            Slider({ value: this.sizeValue, min: 2, max: 256, step: 1 }).onChange((value: number, mode: SliderChangeMode) => {
              this.sizeValue = value;
              if (mode === SliderChangeMode.End || mode === SliderChangeMode.Click) this.send({ type: 'size', axis: this.sizeAxis, value: value });
            })
            Row({ space: 12 }) {
              Button('−').onClick(() => { this.sizeValue = Math.max(2, this.sizeValue - 1); this.send({ type: 'size', axis: this.sizeAxis, value: this.sizeValue }); })
              Button('+').onClick(() => { this.sizeValue = Math.min(256, this.sizeValue + 1); this.send({ type: 'size', axis: this.sizeAxis, value: this.sizeValue }); })
              Button('Done').onClick(() => this.send({ type: 'dismissSize' }))
            }
          }.padding(20).width(300).backgroundColor('#F0242E40').borderRadius(16).align(Alignment.Center)
        }
      }
      if (this.localError.length > 0 || this.stats.error.length > 0) {
        Column({ space: 12 }) {
          Text(this.localError || this.stats.error).fontColor('#FFD0D0').fontSize(14)
          Button('Close').onClick(() => { this.localError = ''; this.send({ type: 'stop' }); this.leave(); })
        }.padding(18).width('90%').backgroundColor('#F0332028').borderRadius(12).align(Alignment.Bottom).margin(16)
      }
    }.width('100%').height('100%').backgroundColor('#161C28')
  }
}
