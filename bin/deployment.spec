# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None


a = Analysis([r"C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\src\main.py"],
             pathex=[
                r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\src',
                r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code',
             ],
             binaries=[
                (r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\.env\Lib\site-packages\pywin32_system32\pythoncom37.dll', '.')],
             datas=[
                (r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\models\DEPLOYMENT', r'models\DEPLOYMENT'),
                (r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\src\static', 'static'),
             ],
             hiddenimports=[
                'pywin32', 'pywin32-ctypes', 'pypiwin32', 'pywinpty', 'win32com', 'cv2', 'pydicom.encoders.gdcm',
                'sklearn.neighbors._partition_nodes', 'h5py'
             ],
             hookspath=[r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\bin\hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name=f'Breast Mass Cancer diagnosis Tool',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          icon=r'C:\Users\USUARIO\Desktop\Master Data Science\TFM\Code\src\static\images\logo.ico'
          )
