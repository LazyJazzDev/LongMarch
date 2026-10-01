import { LongMarchAbility } from 'longmarch';
import BuildProfile from 'BuildProfile';

// The page and native host live in the longmarch HAR; the gameoflife product
// opens its game directly (STANDALONE_GAME in build-profile.json5).
export default class EntryAbility extends LongMarchAbility {
  protected standaloneGame(): string { return BuildProfile.STANDALONE_GAME; }
}
