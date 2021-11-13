from glob import glob
from pathlib import Path
from typing import io, Union

import os


def get_filename(x: io) -> str:
    return os.path.basename(os.path.splitext(x)[0])


def get_dirname(x: io) -> str:
    return os.path.dirname(os.path.splitext(x)[0])


def get_path(*args: Union[io, str], create: bool = True) -> io:
    path = os.path.join(*args)
    if create:
        create_dir(get_dirname(path))
    return path


def create_dir(path: io):
    Path(path).mkdir(parents=True, exist_ok=True)


def search_files(file: io, ext: str) -> iter:
    return glob(os.path.join(file, '**', f'*.{ext}'), recursive=True)