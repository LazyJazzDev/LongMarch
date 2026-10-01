import { resourceManager } from '@kit.LocalizationKit';
export const initialize: (resources: resourceManager.ResourceManager, filesDir: string) => void;
export const command: (json: string) => void;
export const status: () => string;
