import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

# IO types
PathLike = Union[str, Path]


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------
    dest_dir: PathLike
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------
    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def delete_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Delete a folder path.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path that will be removed.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------
    shutil.Error:
        If the folder could not be removed.

    Returns
    ------------------------
        None

    """
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            if verbose:
                print(f"Folder {dest_dir} was removed!")
        except shutil.Error as e:
            print(f"Folder could not be removed! Error {e}")


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
) -> None:

    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(config: dict) -> None:
    """
    Execute a shell command with a given configuration.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    for out in execute_command_helper(
        config["command"], config["verbose"], config["stdout_log_file"]
    ):
        if len(out):
            config["logger"].info(out)

        if config["exists_stdout"]:
            save_string_to_txt(out, config["stdout_log_file"], "a")


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------
    obj: object
        Object that wants to be validated.

    Returns
    ------------------------
    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------
    filename: str
        Name of the json file.
    dictionary: dict
        Dictionary that will be saved as json.
    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary == None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")


def read_json_as_dict(filepath: str) -> dict:

    """
    Reads a json as dictionary.

    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def helper_build_param_value_command(
    params: dict, equal_con: Optional[bool] = True
) -> str:
    """
    Helper function to build a command based on key:value pairs.

    Parameters
    ------------------------
    params: dict
        Dictionary with key:value pairs used for building the command.
    equal_con: Optional[bool]
        Indicates if the parameter is followed by '='. Default True.
    Returns
    ------------------------
    str:
        String with the parameters.

    """
    equal = " "
    if equal_con:
        equal = "="

    parameters = ""
    for (param, value) in params.items():
        if type(value) in [str, float, int] or check_path_instance(value):
            parameters += f"--{param}{equal}{str(value)} "

    return parameters


def helper_additional_params_command(params: List[str]) -> str:
    """
    Helper function to build a command based on values.

    Parameters
    ------------------------
    params: list
        List with additional command values used.

    Returns
    ------------------------
    str:
        String with the parameters.

    """
    additional_params = ""
    for param in params:
        additional_params += f"--{param} "

    return additional_params


def gscfuse_mount(bucket_name: PathLike, params: dict) -> None:
    """
    Mounts a bucket in a GCP Virtual Machine using GCSFUSE.

    Parameters
    ------------------------
    bucket_name: str
        Name of the bucket.

    params: dict
        Dictionary with the GCSFUSE params.

    """

    built_params = helper_build_param_value_command(params, equal_con=False)
    additional_params = helper_additional_params_command(
        params["additional_params"]
    )

    gfuse_cmd = f"gcsfuse {additional_params} {built_params} {bucket_name} {bucket_name}"

    for out in execute_command_helper(gfuse_cmd, True):
        print(out)


def gscfuse_unmount(mount_dir: PathLike) -> None:
    """
    Unmounts a bucket in a VM's local folder.

    Parameters
    ------------------------
    bucket_name: str
        Name of the bucket.

    """

    fuser_cmd = f"fusermount -u {mount_dir}"

    for out in execute_command_helper(fuser_cmd, True):
        print(out)


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def get_deepest_dirpath(
    folder: PathLike, ignore_folders: List[str] = ["metadata"]
) -> PathLike:
    """
    Returns the deepest folder path in the provided folder.

    Parameters
    ------------------------
    folder: PathLike
        Path where the search will be carried out.

    ignore_folders: List[str]
        List of folders that need to be ignored

    Returns
    ------------------------
    PathLike:
        Path of the deepest directory
    """

    deepest_path = None
    deep_val = 0

    for root, dirs, files in os.walk(folder, topdown=False):

        if any(ignore_folder in root for ignore_folder in ignore_folders):
            continue

        for foldername in dirs:
            tmp_path = os.path.join(root, foldername)
            if tmp_path.count(os.path.sep) > deep_val and not any(
                ignore_folder in foldername for ignore_folder in ignore_folders
            ):
                deepest_path = tmp_path
                deep_val = tmp_path.count(os.path.sep)

    return Path(deepest_path)


def generate_data_description(
    input_folder: PathLike, tools: List[dict]
) -> dict:
    """
    Generates data description for the output folder.

    Parameters
    ------------------------
    input_folder: PathLike
        Path where the data is located.

    tools: List[dict]
        List with the used tools in the pipeline.

    Returns
    ------------------------
    dict:
        Dictionary with the data description
    """

    if type(input_folder) == str:
        input_folder = Path(input_folder)

    separator = "/"

    if platform.system() == "Windows":
        separator = "\\"

    splitted_path = list(input_folder.parts)
    root_name = splitted_path[-2]
    root_folder = splitted_path[0:-1]
    root_folder = separator.join(root_folder)

    data_description = glob(root_folder + "/data_description.json")

    if len(data_description):
        data_description = read_json_as_dict(data_description[0])

        data_description["Name"] = f"{data_description['Name']}_stitched"
        data_description["DatasetType"] = "derived"
        data_description["GeneratedBy"] = tools

    else:
        data_description = {}

        data_description["Name"] = f"{root_name}_stitched"
        data_description["DatasetType"] = "derived"
        data_description["License"] = "CC-BY-4.0"
        data_description["Institute"] = "Allen Institute For Neural Dynamics"
        data_description["Group"] = ""
        data_description["Project"] = ""
        data_description["GeneratedBy"] = tools

    return data_description


def check_type_helper(value: Any, val_type: type) -> bool:
    """
    Checks if a value belongs to a specific type.

    Parameters
    ------------------------
    value: Any
        variable data.

    val_type: type
        Type that we want to check.

    Returns
    ------------------------
    bool:
        True if the type is what we expect from the variable data, False otherwise.
    """

    if type(value) != val_type:
        return False

    return True


def generate_timestamp(time_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Generates a timestamp in string format.

    Parameters
    ------------------------
    time_format: str
        String following the conventions to generate the timestamp (https://strftime.org/).

    Returns
    ------------------------
    str:
        String with the actual datetime moment in string format.
    """
    return datetime.now().strftime(time_format)
