import pickle as pk


def create_artifact(file, directory, filename, extension="pkl"):
    """
    Function that create the artifact file to be saved in mlflow
    Args:
        file (byte): the file binary
        directory (str): the temporary directory where to save the file
        filename (str): the name of the file to save
        extension (str): the extension of the file to save

    Returns: the full path in the temporary directory to save

    """
    dir_path = directory.replace("\\", "/")
    file_path = dir_path + "/" + filename + f".{extension}"
    if extension != "pkl":
        with open(file_path, "wb") as f:
            f.write(file)
    else:
        pk.dump(file, open(file_path, "wb"))
    return file_path
