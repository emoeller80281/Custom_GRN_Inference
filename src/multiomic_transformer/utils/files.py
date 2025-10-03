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
        
