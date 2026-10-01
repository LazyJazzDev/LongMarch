"""Complete the Ninja-generated iOS bundle without Xcode build-variable expansion."""
from pathlib import Path
import plistlib
import sys

bundle = Path(sys.argv[1])
path = bundle / 'Info.plist'
info = plistlib.loads(path.read_bytes())
info.update(plistlib.loads(Path(sys.argv[2]).read_bytes()))
# Executable name, bundle identifier and iPhone + iPad (TARGETED_DEVICE_FAMILY 1,2);
# without the device family iPad runs the app in an iPhone-sized window.
info.update(CFBundleExecutable=sys.argv[3], CFBundleIdentifier=sys.argv[4], MinimumOSVersion='18.0',
            UIDeviceFamily=[1, 2])
path.write_bytes(plistlib.dumps(info))
