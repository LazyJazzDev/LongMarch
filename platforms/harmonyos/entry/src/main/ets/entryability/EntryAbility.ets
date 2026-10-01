import { UIAbility, Want, ConfigurationConstant } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import native from 'liblongmarch.so';
import BuildProfile from 'BuildProfile';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want): void {
    this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_DARK);
    // A standalone game product always opens its game; LongMarch Demos opens
    // the requested demo of a smoke launch, if any.
    const standalone: string = BuildProfile.STANDALONE_GAME;
    AppStorage.setOrCreate('longmarchSmokeDemo', standalone.length > 0 ? standalone : String(want.parameters?.demo ?? ''));
    AppStorage.setOrCreate('longmarchSmokeScene', String(want.parameters?.scene ?? ''));
    AppStorage.setOrCreate('longmarchSmokeSamples', Number(want.parameters?.samples ?? 32));
  }
  onWindowStageCreate(stage: window.WindowStage): void {
    try {
      native.initialize(this.context.resourceManager, this.context.filesDir);
    } catch (e) {
      AppStorage.setOrCreate('longmarchStartupError', `Initialization: ${e}`);
    }
    this.configureWindow(stage);
  }
  private async configureWindow(stage: window.WindowStage): Promise<void> {
    try {
      const win = await stage.getMainWindow();
      await win.setWindowLayoutFullScreen(true);
      await win.setWindowSystemBarEnable([]);
      await stage.loadContent('pages/Index');
    } catch (e) {
      AppStorage.setOrCreate('longmarchStartupError', `Window setup: ${e}`);
    }
  }
  onForeground(): void { native.command('{"type":"active","value":true}'); }
  onBackground(): void { native.command('{"type":"active","value":false}'); }
  onWindowStageDestroy(): void { native.command('{"type":"stop"}'); }
}
