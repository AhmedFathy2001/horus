# -*- mode: python ; coding: utf-8 -*-
import os, sys
from PyInstaller.utils.hooks import collect_submodules

IS_MAC = sys.platform == 'darwin'
ICON_MAC = 'assets/horus.icns' if os.path.exists('assets/horus.icns') else None
ICON_WIN = 'assets/horus.ico' if os.path.exists('assets/horus.ico') else None

hiddenimports = []
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('pygame')
hiddenimports += collect_submodules('simpy')


a = Analysis(
    ['run_demo.py'],
    pathex=[],
    binaries=[],
    datas=[('data', 'data'), ('assets', 'assets')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='horus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_WIN,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='horus',
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name='HORUS.app',
        icon=ICON_MAC,
        bundle_identifier='com.pantomath.horus',
        info_plist={
            'CFBundleName': 'HORUS',
            'CFBundleDisplayName': 'HORUS',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSHumanReadableCopyright': 'Pantomath · Hackatom 2026',
        },
    )
