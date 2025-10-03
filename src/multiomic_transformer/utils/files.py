import os
import json
from pathlib import Path

def unique_path(directory: Path, name_pattern: str):
    """
    Returns a path for the next iteration in a directory naming pattern
    
    e.g. if `directory` contains the subdirectories `test_001` and `test_002`,
    then the template `test_{:03d}` will return `test_003`.

    Parameters
    -----------
        directory (Path)
            A `Path` object for the target directory.
        name_pattern (str)
            An iterable pattern for the iteration.
    
    Returns
    ----------
        path (Path)
        Path for the next iteration in the pattern.
    
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path
        
def atomic_json_dump(obj, path: Path):
    """Safe JSON dump, avoids race conditions by making a tmp file first, then updating the name"""
    def convert(o):
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=convert)
    os.replace(tmp, path)