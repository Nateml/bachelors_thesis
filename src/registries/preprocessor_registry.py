from src.data.preprocessing import bandpass, highpass, median_filter


def get_preprocessor(name: str):
    if name == "bandpass":
        return bandpass
    elif name == "highpass":
        return highpass
    elif name == "median":
        median_filter
    else:
        raise ValueError(f"Unknown preprocessor: {name}")