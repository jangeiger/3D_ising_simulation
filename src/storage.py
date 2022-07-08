import pickle  # for input/output




def save_data(filename, data, folder="data"):
    """Save an (almost) arbitrary python object to disc."""
    with open(folder+"/"+filename, 'wb') as f:
        pickle.dump(data, f)
    # done


def load_data(filename, folder="data"):
    """Load and return data saved to disc with the function `save_data`."""
    with open(folder+"/"+filename, 'rb') as f:
        data = pickle.load(f)
    return data