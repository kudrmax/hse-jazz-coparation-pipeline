"""BebopNet model package (lazy imports to avoid torch dependency in test contexts)."""

__all__ = ["GeneratorBebopnet", "GeneratorBebopnetInput", "GeneratorBebopnetOutput"]


def __getattr__(name: str):
    """Lazy-load model classes only when accessed (torch not required for tests)."""
    if name == "GeneratorBebopnet":
        from .generator import GeneratorBebopnet
        return GeneratorBebopnet
    elif name == "GeneratorBebopnetInput":
        from .input import GeneratorBebopnetInput
        return GeneratorBebopnetInput
    elif name == "GeneratorBebopnetOutput":
        from .output import GeneratorBebopnetOutput
        return GeneratorBebopnetOutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
