from bachelors_thesis.data.preprocessing import bandpass, highpass


def get_preprocessor(name: str):
    if name == "bandpass":
        return bandpass
    elif name == "highpass":
        return highpass
    else:
        raise ValueError(f"Unknown preprocessor: {name}")