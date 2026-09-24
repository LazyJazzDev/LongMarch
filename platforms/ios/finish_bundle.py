"""Complete the Ninja-generated iOS bundle without Xcode build-variable expansion."""
from pathlib import Path
import plistlib
import sys

bundle = Path(sys.argv[1])
path = bundle / 'Info.plist'
info = plistlib.loads(path.read_bytes())
info.update(plistlib.loads(Path(sys.argv[2]).read_bytes()))
info.update(CFBundleExecutable='LongMarch', CFBundleIdentifier='dev.lazyjazz.longmarch', MinimumOSVersion='18.0')
path.write_bytes(plistlib.dumps(info))
