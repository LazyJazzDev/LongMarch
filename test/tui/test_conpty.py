"""Run with tui_menu_probe.exe and optionally a bundled conpty.dll path.

Exercises the real ConPTY stream, unlike native WriteConsoleInput event injection.
Only the probe is spawned; no visible window or user terminal is modified.
"""
import ctypes as c
from ctypes import wintypes as w
import pathlib
import sys
import threading
import time
PROBE = str(pathlib.Path(sys.argv[1]).resolve())
conpty = c.WinDLL(sys.argv[2], use_last_error=True) if len(sys.argv) > 2 else c.WinDLL('kernel32', use_last_error=True)
k = c.WinDLL('kernel32', use_last_error=True)
ptr = c.c_void_p

class Coord(c.Structure):
    _fields_ = [('x', w.SHORT), ('y', w.SHORT)]

class SI(c.Structure):
    _fields_ = [('cb', w.DWORD),
                ('reserved', w.LPWSTR),
                ('desktop', w.LPWSTR),
                ('title', w.LPWSTR),
                ('x', w.DWORD),
                ('y', w.DWORD),
                ('cx', w.DWORD),
                ('cy', w.DWORD),
                ('cols', w.DWORD),
                ('rows', w.DWORD),
                ('fill', w.DWORD),
                ('flags', w.DWORD),
                ('show', w.WORD),
                ('reserved2', w.WORD),
                ('bytes', ptr),
                ('stdin', w.HANDLE),
                ('stdout', w.HANDLE),
                ('stderr', w.HANDLE)]

class SIX(c.Structure):
    _fields_ = [('si', SI), ('attrs', ptr)]

class PI(c.Structure):
    _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]
k.CreatePipe.argtypes = [c.POINTER(w.HANDLE), c.POINTER(w.HANDLE), ptr, w.DWORD]
conpty.CreatePseudoConsole.argtypes = [Coord, w.HANDLE, w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
conpty.CreatePseudoConsole.restype = c.c_long
k.InitializeProcThreadAttributeList.argtypes = [ptr, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)]
k.UpdateProcThreadAttribute.argtypes = [ptr, w.DWORD, c.c_size_t, ptr, c.c_size_t, ptr, ptr]
k.CreateProcessW.argtypes = [w.LPCWSTR, w.LPWSTR, ptr, ptr, w.BOOL, w.DWORD, ptr, w.LPCWSTR, c.POINTER(SIX), c.POINTER(PI)]
k.ReadFile.argtypes = [w.HANDLE, ptr, w.DWORD, c.POINTER(w.DWORD), ptr]
k.WriteFile.argtypes = k.ReadFile.argtypes
conpty.ResizePseudoConsole.argtypes = [w.HANDLE, Coord]
conpty.ResizePseudoConsole.restype = c.c_long
k.CloseHandle.argtypes = [w.HANDLE]
conpty.ClosePseudoConsole.argtypes = [w.HANDLE]
k.DeleteProcThreadAttributeList.argtypes = [ptr]
k.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]

def check(x):
    if not x:
        raise c.WinError(c.get_last_error())
ir, iw, orr, ow = [w.HANDLE() for _ in range(4)]
check(k.CreatePipe(c.byref(ir), c.byref(iw), None, 0))
check(k.CreatePipe(c.byref(orr), c.byref(ow), None, 0))
hpc = w.HANDLE()
assert conpty.CreatePseudoConsole(Coord(80, 24), ir, ow, 0, c.byref(hpc)) == 0
k.CloseHandle(ir)
k.CloseHandle(ow)
size = c.c_size_t()
k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
attrs = c.create_string_buffer(size.value)
check(k.InitializeProcThreadAttributeList(attrs, 1, 0, c.byref(size)))
check(k.UpdateProcThreadAttribute(attrs, 0, 0x00020016, hpc, c.sizeof(hpc), None, None))
k.GetStdHandle.restype = w.HANDLE
k.SetStdHandle.argtypes = [w.DWORD, w.HANDLE]
saved = [k.GetStdHandle(i) for i in (-10, -11, -12)]
# Prevent this driver's redirected handles from leaking into the pseudoconsole child.
for i in (-10, -11, -12):
    k.SetStdHandle(i, None)
s = SIX()
s.si.cb = c.sizeof(s)
s.attrs = c.cast(attrs, ptr)
pi = PI()
exe = PROBE
check(k.CreateProcessW(None, c.create_unicode_buffer('"' + exe + '"'), None, None, False, 0x00080000, None, None, c.byref(s), c.byref(pi)))
for i, h in zip((-10, -11, -12), saved):
    k.SetStdHandle(i, h)
k.CloseHandle(pi.thread)
k.DeleteProcThreadAttributeList(attrs)
data = bytearray()

def reader():
    while True:
        buf = c.create_string_buffer(4096)
        n = w.DWORD()
        if not k.ReadFile(orr, buf, 4096, c.byref(n), None) or not n.value:
            break
        data.extend(buf.raw[:n.value])
t = threading.Thread(target=reader, daemon=True)
t.start()
try:
    deadline = time.monotonic() + 5
    while b'TUI integration probe' not in data and time.monotonic() < deadline:
        time.sleep(0.02)
    assert b'TUI integration probe' in data, 'Menu did not open in ConPTY'
    assert b'\x1b[?1000h' in data, 'Outer terminal mouse capture was not enabled'
    assert b'\x1b[?1006h' in data, 'SGR mouse coordinates were not enabled'
    n = w.DWORD()
    for chunk in (b'\x1b[<0;3;5M', b'\x1b[<0;3;5m', b'\x1b[<65;3;5M'):
        check(k.WriteFile(iw, chunk, len(chunk), c.byref(n), None))
        time.sleep(0.02)
    assert conpty.ResizePseudoConsole(hpc, Coord(30, 2)) == 0
    time.sleep(0.1)
    final_keys = b'\x1b[B\x1b[B\r'
    check(k.WriteFile(iw, final_keys, len(final_keys), c.byref(n), None))
    assert k.WaitForSingleObject(pi.process, 5000) == 0, 'Menu did not complete'
    result = w.DWORD()
    check(k.GetExitCodeProcess(pi.process, c.byref(result)))
    assert result.value == 15, f'Expected item5 after click, wheel, resize and two Down keys; exit={result.value}'
finally:
    conpty.ClosePseudoConsole(hpc)
    t.join(2)
    k.CloseHandle(iw)
    k.CloseHandle(orr)
    k.CloseHandle(pi.process)
assert b'\x1b[?1006l' in data, 'SGR mouse mode was not restored'
assert b'\x1b[?1000l' in data, 'Mouse capture was not disabled'
print('ConPTY: mouse enable/disable, SGR click, wheel, two-row resize, arrow keys, Enter and terminal restoration passed')
