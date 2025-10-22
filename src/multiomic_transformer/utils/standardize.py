def standardize_name(name: str) -> str:
    """Convert gene/motif name to capitalization style (e.g. 'Hoxa2')."""
    if not isinstance(name, str):
        return name
    return name.upper()