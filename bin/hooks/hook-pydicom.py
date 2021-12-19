from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('pydicom.encoders')
hiddenimports.extend(collect_submodules('pydicom.overlays'))
hiddenimports.extend(collect_submodules('pydicom.overlay_data_handlers'))