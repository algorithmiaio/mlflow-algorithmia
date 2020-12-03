import os
import subprocess

import Algorithmia


client = Algorithmia.client()

api_key = os.environ.get("ALGO_API_KEY", None)
if api_key is not None:
    client = Algorithmia.client(api_key)
else:
    client = Algorithmia.client()


in_algorithmia = True if os.environ.get("ALGORITHMIA_API", False) else False


def extract_tar_gz(file, output_dir="./models"):
    """
    Extract a .tar.gz
    Parameters
    ----------
        output_dir (default="./models"): Where to extract the .tar.gz
    Returns
    ------
        output_dir: full path to the output directory where files where extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        output = subprocess.check_output(
            "tar -C {output} -xzf {targz}".format(output=output_dir, targz=file),
            stderr=subprocess.STDOUT,
            shell=True,
        ).decode()
    except subprocess.CalledProcessError as ex:
        output = ex.output.decode()
        raise Exception("Could not extract file: %s" % output)

    return os.path.realpath(os.path.join(output_dir))


def get_file(remote_fpath):
    """
    Download a file hosted on Algorithmia Hosted Data
    If the file ends with .tar.gz it will untar the file.
    It's recommended that the tar file contain a single files compressed like:
        tar -czvf model.format.tar.gz model.format
    Returns the local file path of the downloaded file
    """
    basename = os.path.basename(remote_fpath)

    if remote_fpath.startswith("data://"):
        # Download from Algoritmia hosted data
        local_fpath = client.file(remote_fpath).getFile().name

        if basename.endswith(".tar.gz"):
            output_dir = extract_tar_gz(local_fpath)
            no_ext = basename[: -len(".tar.gz")]
            local_fpath = os.path.join(output_dir, no_ext)

        return local_fpath

    return remote_fpath


def exists(username, collection, fname=None, connector="data"):
    if fname is None:
        path = f"{connector}://{username}/{collection}"
        obj = client.dir(path)
        return obj.exists()
    else:
        path = f"{connector}://{username}/{collection}/{fname}"
        obj = client.file(path)
        return obj.exists()


def upload_file(
    local_filename,
    username,
    collection,
    fname,
    connector="data",
):
    dir_exists = exists(username=username, collection=collection, connector=connector)
    if dir_exists is False:
        dir_path = f"{connector}://{username}/{collection}/"
        new_dir = client.dir(dir_path)
        new_dir.create()

    remote_file = f"{connector}://{username}/{collection}/{fname}"
    client.file(remote_file).putFile(local_filename)
    return remote_file
