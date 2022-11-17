import errno
import logging
import math
import os
import platform
import re
import subprocess
import sys
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

from argschema import ArgSchemaParser
from ng_link import NgState
from utils import utils
from zarr_converter import ZarrConverter

from .params import PipelineParams, get_default_config
from .path_parser import PathParser

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")

PathLike = Union[str, Path]


class TeraStitcher:
    def __init__(
        self,
        input_data: PathLike,
        output_folder: PathLike,
        parallel: bool = True,
        pyscripts_path: Optional[PathLike] = None,
        computation: Optional[str] = "cpu",
        preprocessing: Optional[dict] = None,
        verbose: Optional[bool] = False,
        gcloud_execution: Optional[bool] = False,
        preprocessing_folder: Optional[PathLike] = None,
    ) -> None:

        """
        Class constructor

        Parameters
        ------------------------
        input_data: PathLike
            Path where the data is stored.
        output_folder: PathLike
            Path where the stitched data will be stored.
        parallel: Optional[bool]
            True if you want to run terastitcher in parallel, False otherwise.
        pyscripts_path: Optional[PathLike]
            Path where parastitcher and paraconverter execution files are located.
        computation: Optional[str]
            String that indicates where will terastitcher run. Available options are: ['cpu', 'gpu']
        preprocessing: Optional[dict]:
            All the preprocessing steps prior to terastitcher's pipeline. Default None.
        verbose: Optional[bool]
            True if you want to print outputs of all executed commands.

        Raises
        ------------------------
        FileNotFoundError:
            If terastitcher, Parastitcher or paraconverter (if provided) were not found in the system.

        """

        self.__input_data = Path(input_data)
        self.__output_folder = Path(output_folder).joinpath("processed")
        self.__output_jsons_path = Path(output_folder)
        self.__preprocessing_folder = Path(preprocessing_folder)
        self.__stitched_folder = self.__preprocessing_folder.joinpath(
            "stitched"
        )
        self.__parallel = parallel
        self.__computation = computation
        self.__platform = platform.system()
        self.__parastitcher_path = Path(pyscripts_path).joinpath(
            "Parastitcher.py"
        )
        self.__paraconverter_path = Path(pyscripts_path).joinpath(
            "paraconverter.py"
        )
        self.preprocessing = preprocessing
        self.__verbose = verbose
        self.__python_terminal = None
        self.metadata_path = self.__output_folder.joinpath("metadata/params")
        self.xmls_path = self.__output_folder.joinpath("metadata/xmls")
        self.ome_zarr_path = self.__output_folder.joinpath("OMEZarr")
        self.gcloud_execution = gcloud_execution

        # Check python
        self.__check_python()

        if computation not in ["cpu", "gpu"]:
            print("Setting computation to cpu")
            self.__computation = "cpu"

        if computation == "gpu":
            # Setting environment variable that terastitcher sees for cuda implementation of MIP-NCC algorithm
            # TODO check if cuda is availabe and the toolkit and drivers are correct
            os.environ["USECUDA_X_NCC"] = "1"

        else:
            try:
                del os.environ["USECUDA_X_NCC"]
            except KeyError:
                warnings.warn(
                    "environmental variable 'USECUDA_X_NCC' could not be removed. Ignore this warning if you're using CPU"
                )

        if not self.__check_installation():
            print(
                f"Please, check your terastitcher installation in the system {self.__platform}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "terastitcher"
            )

        # TODO get versions
        tools = [
            {
                "Name": "TeraStitcher",
                "Version": "1.11.10",
                "CodeURL": "https://github.com/camilolaiton/TeraStitcher.git@fix/data_paths",
            },
            {
                "Name": "aicsimageio",
                "Version": "feature/zarrwriter-multiscales-daskjobs",
                "CodeURL": "https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs",
            },
        ]

        pystripe_info = {
            "Name": "pystripe",
            "Version": "0.2.0",
            "CodeURL": "https://github.com/camilolaiton/pystripe.git@feature/output_format",
        }

        data_description = utils.generate_data_description(
            input_folder=self.__input_data, tools=tools
        )
        data_description_path = self.__output_jsons_path.joinpath(
            "data_description.json"
        )

        # If parastitcher or paraconverter paths are not found, we set computation to sequential cpu as default.
        self.__check_teras_parallel_scripts()

        # We create the folders for the xmls and metadata in the output directory
        utils.create_folder(self.xmls_path, self.__verbose)
        utils.create_folder(self.metadata_path, self.__verbose)
        utils.create_folder(self.__stitched_folder)

        # Setting stdout log file last because the folder structure depends if preprocessing steps are provided
        self.stdout_log_file = self.metadata_path.joinpath("stdout_log.txt")

        # Saving data description

        if preprocessing and "pystripe" in preprocessing:
            tools.insert(0, pystripe_info)

        utils.save_dict_as_json(data_description_path, data_description)

        # Setting logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def __check_installation(self, tool_name: str = "terastitcher") -> bool:
        """
        Checks the installation of any tool in the system environment.

        Parameters
        ------------------------
        tool_name: str
            command name to check the installation. Default: 'terastitcher'

        Returns
        ------------------------
        bool:
            True if the command was correctly executed, False otherwise.

        """

        try:
            devnull = open(os.devnull)
            subprocess.Popen(
                [tool_name], stdout=devnull, stderr=devnull
            ).communicate()
        except OSError as e:
            return False
        return True

    def __check_python(self) -> None:
        """
        Checks python3 installation in the system.

        Parameters
        ------------------------
        None

        Raises
        ------------------------
        FileNotFoundError:
            If python was not found in the system.

        Returns
        ------------------------
        None

        """

        def helper_status_cmd(cmd: List[str]) -> int:
            """
            Helper function to check python terminal execution.

            Parameters
            ------------------------
            cmd: List[str]
                command splitted in list mode.

            Returns
            ------------------------
            int:
                Process exit status.
            """
            exit_status = None

            try:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                exit_status = proc.returncode

            except FileNotFoundError as err:
                exit_status = -1

            return exit_status

        found = True
        if sys.version_info.major == 3:

            if not helper_status_cmd(["python", "-V"]):
                self.__python_terminal = "python"

            elif not helper_status_cmd(["python3", "-V"]):
                self.__python_terminal = "python3"

            else:
                found = False
        else:
            found = False

        if not found:
            self.logger.info(
                f"Please, check your python 3 installation in the system {self.__platform}"
            )
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "python"
            )

    def __check_teras_parallel_scripts(self) -> None:
        """
        Checks parastitcher and paraconverter installation using a provided paths.

        Raises
        ------------------------
        FileNotFoundError:
            If parastitcher or paraconverter were not found in the system.

        """

        def check_file_helper(path: PathLike) -> bool:
            if path != None:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} not found.")

                return True
            return False

        parastitcher = check_file_helper(self.__parastitcher_path)
        paraconverter = check_file_helper(self.__paraconverter_path)

        if not parastitcher or not paraconverter:
            self.__parallel = False

    def __build_parallel_command(
        self, params: dict, step_name: str, tool: PathLike
    ) -> str:
        """
        Builds a mpi command based on a provided configuration dictionary.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the mpi command depending on the platform.
        step_name: str
            Terastitcher runs in parallel the align and merge steps. Then, we build the command
            based on which step terastitcher is running.
        tool: PathLike
            Parallel tool to be used in the command. (Parastitcher or Paraconverter)

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """

        if not self.__parallel:
            return ""

        cpu_params = params["cpu_params"]

        # mpiexec for windows, mpirun for linux or macs OS
        mpi_command = (
            "mpiexec -n" if self.__platform == "Windows" else "mpirun -np"
        )
        additional_params = ""
        hostfile = ""
        n_procs = cpu_params["number_processes"]

        # Additional params provided in the configuration
        if (
            len(cpu_params["additional_params"])
            and self.__platform != "Windows"
        ):
            additional_params = utils.helper_additional_params_command(
                cpu_params["additional_params"]
            )

        # Windows does not require a hostfile to work
        if self.__platform != "Windows":
            try:
                hostfile = f"--hostfile {cpu_params['hostfile']}"
            except KeyError:
                self.logger.info(
                    "Hostfile was not found. This could lead to execution problems."
                )

        # If we want to estimate the number of processes used in any of the steps.
        if cpu_params["estimate_processes"]:
            if step_name == "align":
                n_procs = TeraStitcher.get_aprox_number_processes_align_step(
                    {
                        "image_depth": cpu_params["image_depth"],
                        "subvoldim": params["subvoldim"],
                        "number_processes": cpu_params["number_processes"],
                    }
                )

                self.logger.info(
                    f"- Changing number of processes for align step to {n_procs}"
                )

            elif step_name == "merge":
                # TODO estimate in merge step
                self.logger.info(
                    "Aproximate number of processes for the merge step is not implemented yet."
                )

        cmd = f"{mpi_command} {n_procs} {hostfile} {additional_params}"
        cmd += f"{self.__python_terminal} {tool}"
        return cmd

    def import_step_cmd(
        self, params: dict, channel: str, fuse_path: PathLike = None
    ) -> str:
        """
        Builds the terastitcher's import command based on a provided configuration dictionary.
        It outputs a json file in the xmls folder of the output directory with all the parameters
        used in this step.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's import command.
        channel:str
            Name of the dataset channel that will be imported
        fuse_path:PathLike
            Path where the fused xml files will be stored. This will only be used in multichannel fusing.
            Default None

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """

        # TODO Check if params comes with all necessary keys so it does not raise KeyNotFound error

        input_path = self.__input_data.joinpath(channel)

        volume_input = f"--volin={input_path}"

        output_path = self.xmls_path.joinpath(f"xml_import_{channel}.xml")

        # changing output path to fuse path for multichannel fusing
        if fuse_path != None:
            output_path = fuse_path.joinpath(f"xml_import_{channel}.xml")

        output_folder = f"--projout={output_path}"

        parameters = utils.helper_build_param_value_command(params)

        additional_params = ""
        if len(params["additional_params"]):
            additional_params = utils.helper_additional_params_command(
                params["additional_params"]
            )

        cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"

        output_json = self.metadata_path.joinpath(
            f"import_params_{channel}.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    def import_multivolume_cmd(self, params: dict) -> str:
        """
        Builds the terastitcher's multivolume import command based on a provided configuration dictionary.
        It outputs a json file in the xmls fuse folder of the output directory with all the parameters
        used in this step.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's import command.

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """

        parameters = utils.helper_build_param_value_command(params)

        additional_params = ""
        if len(params["additional_params"]):
            additional_params = utils.helper_additional_params_command(
                params["additional_params"]
            )

        cmd = f"terastitcher --import {parameters} {additional_params}"

        output_json = self.metadata_path.joinpath(
            f"import_params_multivolume.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    @staticmethod
    def get_aprox_number_processes_align_step(config_params: dict) -> int:
        """
        Get the estimate number of processes to partition the dataset and calculate the align step.
        Using MPI, check if the number of slots are enough for running the number of processes.
        You can automatically set --use-hwthread-cpus to automatically estimate the number of
        hardware threads in each core and increase the allowed number of processes. There is
        another option with -oversubscribe.

        Parameters:
        -----------------
        config_params: dict
            Parameters that will be used in the align step.
            i.e. {'image_depth': 4200, 'subvoldim': 100, 'number_processes': 10}

        Returns:
        -----------------
        int:
            Number of processes to be used in the align step. If it is not possible to perform
            the estimation, we return 2 processes as default (the master and slave processes).

        """

        if (
            config_params["image_depth"] < config_params["number_processes"]
            or config_params["subvoldim"] > config_params["image_depth"]
        ):
            print(
                "Please check the parameters for aproximate number of processes in align step"
            )
            return 2

        # Partitioning depth for the tiles
        partitioning_depth = math.ceil(
            config_params["image_depth"] / config_params["subvoldim"]
        )

        for proc in range(config_params["number_processes"], 0, -1):
            tiling_proc = 2 * (proc - 1)

            if partitioning_depth > tiling_proc:
                return proc

        return 2

    def align_step_cmd(self, params: dict, channel: str) -> str:
        """
        Builds the terastitcher's align command based on a provided configuration dictionary.
        It outputs a json file in the xmls folder of the output directory with all the parameters
        used in this step.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's align command.
        channel:str
            Name of the dataset channel that will be aligned

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """
        input_path = self.xmls_path.joinpath(f"xml_import_{channel}.xml")
        input_xml = f"--projin={input_path}"

        output_path = self.xmls_path.joinpath(f"xml_displcomp_{channel}.xml")
        output_xml = f"--projout={output_path}"
        parallel_command = ""

        if self.__parallel:
            parallel_command = self.__build_parallel_command(
                params, "align", self.__parastitcher_path
            )

        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"

        parameters = utils.helper_build_param_value_command(params)

        cmd = f"{parallel_command} --displcompute {input_xml} {output_xml} {parameters} > {self.xmls_path}/step2par.txt"

        output_json = self.metadata_path.joinpath(
            f"align_params_{channel}.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    def input_output_step_cmd(
        self,
        step_name: str,
        input_xml: str,
        output_xml: str,
        channel: str,
        params: Optional[dict] = None,
    ) -> str:

        """
        Builds the terastitcher's input-output commands based on a provided configuration dictionary.
        These commands are: displproj for projection, displthres for threshold and placetiles for
        placing tiles. Additionally, it outputs a json file in the xmls folder of the output directory
        with all the parameters used in this step.

        Parameters
        ------------------------
        step_name: str
            Name of the step that will be executed. The names should be: 'displproj' for projection,
            'displthres' for threshold and 'placetiles' for placing tiles step.
        input_xml: str
            The xml filename outputed from the previous command.
        output_xml: str
            The xml filename that will be used as output for this step.
        params: dict
            Configuration dictionary used to build the terastitcher's command.

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """

        input_path = self.xmls_path.joinpath(input_xml)
        input_xml = f"--projin={input_path}"

        output_path = self.xmls_path.joinpath(output_xml)
        output_xml = f"--projout={output_path}"

        parameters = ""

        if params:
            parameters = utils.helper_build_param_value_command(params)

        cmd = (
            f"terastitcher --{step_name} {input_xml} {output_xml} {parameters}"
        )

        output_json = self.metadata_path.joinpath(
            f"{step_name}_params_{channel}.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    def merge_step_cmd(self, params: dict, channel: str) -> str:
        """
        Builds the terastitcher's merge command based on a provided configuration dictionary.
        It outputs a json file in the xmls folder of the output directory with all the parameters
        used in this step.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's merge command.
        channel:str
            Name of the dataset channel that will be merged

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """

        input_path = self.xmls_path.joinpath(f"xml_merging_{channel}.xml")
        input_xml = f"--projin={input_path}"

        output_path = f"--volout={self.__stitched_folder}"
        parallel_command = ""

        params = {
            "slicewidth": params["slice_extent"][0],
            "sliceheight": params["slice_extent"][1],
            "slicedepth": params["slice_extent"][2],
            "volout_plugin": params["volout_plugin"],
            "cpu_params": params["cpu_params"],
        }

        if self.__parallel:
            parallel_command = self.__build_parallel_command(
                params, "merge", self.__parastitcher_path
            )

        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"terastitcher"

        parameters = utils.helper_build_param_value_command(params)

        cmd = f"{parallel_command} --merge {input_xml} {output_path} {parameters} > {self.xmls_path}/step6par.txt"

        output_json = self.metadata_path.joinpath(
            f"merge_params_{channel}.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    def merge_multivolume_cmd(self, params: dict) -> str:
        """
        Builds the terastitcher's multivolume merge command based on a provided configuration dictionary.
        It outputs a json file in the xmls folder of the output directory with all the parameters
        used in this step.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the terastitcher's multivolume merge command.

        Returns
        ------------------------
        str:
            Command that will be executed for terastitcher.

        """
        parallel_command = ""

        if self.__parallel:
            parallel_command = self.__build_parallel_command(
                params, "merge", self.__paraconverter_path
            )

        else:
            # Sequential execution or gpu execution if USECUDA_X_NCC flag is 1
            parallel_command = f"teraconverter"

        parameters = utils.helper_build_param_value_command(params)

        cmd = f"{parallel_command} {parameters}"  # > {self.xmls_path}/step6par.txt"
        cmd = cmd.replace("--s=", "-s=")
        cmd = cmd.replace("--d=", "-d=")

        output_json = self.metadata_path.joinpath("merge_volume_params.json")
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        return cmd

    def convert_to_ome_zarr(self, config: dict, channels: List[str]) -> None:
        """
        Converts tiff files to OMEZarr.

        Parameters
        ------------------------
        config: dict
            Configuration dictionary used to instanciate the OMEZarr Writer.

        channels:List[str]
            List with channel names that will be processed by pystripe.
            Sigma1 and sigma2 values in the lists belong to each of the channels respectively.
        """

        output_json = self.metadata_path.joinpath("ome_zarr_params.json")
        utils.save_dict_as_json(output_json, config, self.__verbose)

        path = [
            folder
            for folder in os.listdir(self.__stitched_folder)
            if "RES" in folder
        ]

        converter = ZarrConverter(
            self.__stitched_folder.joinpath(path[0]),
            self.ome_zarr_path,
            {"codec": config["codec"], "clevel": config["clevel"]},
            channels,
            config["physical_pixels"],
        )

        converter.convert(config)

    def create_ng_link(self, config: dict, channels: List[str]) -> None:

        dimensions = {
            "z": {
                "voxel_size": config["physical_pixels"][0],
                "unit": "microns",
            },
            "y": {
                "voxel_size": config["physical_pixels"][1],
                "unit": "microns",
            },
            "x": {
                "voxel_size": config["physical_pixels"][2],
                "unit": "microns",
            },
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        }

        colors = ["red", "green", "purple", "yellow"]

        # Creating layer per channel
        layers = []
        for channel_idx in range(len(channels)):
            layers.append(
                {
                    "source": self.ome_zarr_path.joinpath(
                        channels[channel_idx] + ".zarr"
                    ),
                    "channel": 0,  # channel_idx, # use channel idx when source is the same in zarr to change channel
                    "name": channels[channel_idx],
                    "shader": {
                        "color": colors[channel_idx],
                        "emitter": "RGB",
                        "vec": "vec3",
                    },
                    "shaderControls": {  # Optional
                        "normalized": {"range": [0, 500]}
                    },
                }
            )

        neuroglancer_link = NgState(
            input_config={"dimensions": dimensions, "layers": layers},
            mount_service=config["mount_service"],
            bucket_path=config["bucket_path"],
            output_json=self.__output_jsons_path,
            base_url=config["ng_base_url"],
        )

        neuroglancer_link.save_state_as_json()
        link = neuroglancer_link.get_url_link()
        self.logger.info(f"Visualization link: {link}")

    def __preprocessing_tool_cmd(
        self, tool_name: str, params: dict, equal_con: bool
    ) -> str:

        """
        Builds the execution command for the given tool.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the command.

        tool_name: str
            Tool name to be used in the terminal for execution.

        equal_con: Optional[bool]
            Indicates if the parameter is followed by '='.

        Returns
        ------------------------
        str:
            Command that will be executed for pystripe.

        """
        parameters = utils.helper_build_param_value_command(
            params, equal_con=equal_con
        )

        output_json = self.metadata_path.joinpath(
            f"{tool_name}_params_{params['input'].stem}.json"
        )
        utils.save_dict_as_json(f"{output_json}", params, self.__verbose)

        cmd = f"{tool_name} {parameters}"

        return cmd

    def __execute_preprocessing_steps(
        self, exec_config: dict, channels: List[str]
    ) -> None:
        """
        Executes any preprocessing steps that are required for the pipeline.
        It is necessary to have the begining of the terminal command as key and
        the parameters as a dictionary. i.e. "pystripe": {"input" : input_data,
        "output" : output_folder,"sigma1" : [256, ...],"sigma2" : [256, ...],"workers" : 8}
        Command line would be: pystripe --input input_data --output output_folder
        --sigma1 256 --sigma2 256 --workers 8

        Parameters
        ------------------------
        exec_config: dict
            Configuration for command line execution. Mostly for logger.
        channels:List[str]
            List with channel names that will be processed by pystripe.
            Sigma1 and sigma2 values in the lists belong to each of the channels respectively.

        """
        for idx in range(len(channels)):
            for tool_name, params in self.preprocessing.items():
                params_copy = params.copy()

                params_copy["input"] = params_copy["input"].joinpath(
                    channels[idx]
                )
                params_copy["output"] = params_copy["output"].joinpath(
                    channels[idx]
                )
                params_copy["sigma1"] = params_copy["sigma1"][idx]
                params_copy["sigma2"] = params_copy["sigma2"][idx]

                exec_config["command"] = self.__preprocessing_tool_cmd(
                    tool_name, params_copy, False
                )
                utils.execute_command(exec_config)

    def __compute_informative_channel(
        self,
        config: dict,
        exec_config: dict,
        informative_channel: str,
        fuse_xmls: PathLike,
    ) -> None:
        """
        Computes 1-5 terastitcher steps for the informative channel when the multichannel stitch will be performed.

        Parameters
        ------------------------
        params: dict
            Configuration dictionary used to build the command.
        exec_config: dict
            Configuration for command line execution. Mostly for logger.
        informative_channel:str
            Name of the dataset's informative channel that will be used for stitching.
        fuse_xmls:PathLike
            Path where the multivolume xmls will be saved.
        """

        # Importing informative channel
        exec_config["command"] = self.import_step_cmd(
            config["import_data"], informative_channel
        )

        self.logger.info("Import step for informative channel...")
        # print(exec_config)
        utils.execute_command(exec_config)

        # Compute alignments for most informative channel
        self.logger.info("Align step...")
        exec_config["command"] = self.align_step_cmd(
            config["align"], informative_channel
        )

        utils.execute_command(exec_config)

        # Step 3
        self.logger.info("Projection step...")
        exec_config["command"] = self.input_output_step_cmd(
            "displproj",
            f"xml_displcomp_{informative_channel}.xml",
            f"xml_displproj_{informative_channel}.xml",
            informative_channel,
        )
        utils.execute_command(exec_config)

        # Step 4
        self.logger.info("Threshold step...")
        threshold_cnf = {
            "threshold": config["threshold"]["reliability_threshold"]
        }
        exec_config["command"] = self.input_output_step_cmd(
            "displthres",
            f"xml_displproj_{informative_channel}.xml",
            f"xml_displthres_{informative_channel}.xml",
            informative_channel,
            threshold_cnf,
        )
        utils.execute_command(exec_config)

        # Step 5
        self.logger.info("Placing tiles step...")
        exec_config["command"] = self.input_output_step_cmd(
            "placetiles",
            f"xml_displthres_{informative_channel}.xml",
            f"{fuse_xmls}/xml_merging_{informative_channel}.xml",
            informative_channel,
        )
        utils.execute_command(exec_config)

    def stitch_multiple_channels(
        self,
        config: dict,
        exec_config: dict,
        channels: List[str],
        pos_informative_channel: int = 0,
    ) -> None:
        """
        Stitch a dataset using multiple channels.

        Parameters
        ------------------------
        config: dict
            Configuration dictionary used to process the channels.
        exec_config: dict
            Configuration for command line execution. Mostly for logger.
        channels:List[str]
            List with channel names that will be processed by pystripe.
            Sigma1 and sigma2 values in the lists belong to each of the channels respectively.
        pos_informative_channel:int
            Position of the channels list where the informative channel is located.
        """

        # Creating fuse folder
        informative_channel = channels[pos_informative_channel]
        fuse_xmls = self.xmls_path.joinpath("fuse_xmls")
        utils.create_folder(fuse_xmls)

        # Importing non-informative channels
        for idx in range(len(channels)):

            if idx == pos_informative_channel:
                # Ignore import informative channel since we have already calculated projections
                continue

            exec_config["command"] = self.import_step_cmd(
                config["import_data"], channels[idx], fuse_path=fuse_xmls
            )
            self.logger.info(f"Import step for {channels[idx]} channel...")
            utils.execute_command(exec_config)

        # Computing xmls for informative channel
        self.__compute_informative_channel(
            config, exec_config, informative_channel, fuse_xmls
        )

        # Importing multivolume dataset
        params_multivolume = config["import_data"].copy()
        params_multivolume["volin"] = fuse_xmls
        params_multivolume["projout"] = fuse_xmls.joinpath(
            "import_multivolume.xml"
        )
        params_multivolume["volin_plugin"] = "MultiVolume"
        params_multivolume["imin_channel"] = pos_informative_channel

        exec_config["command"] = self.import_multivolume_cmd(
            params_multivolume
        )
        self.logger.info(
            f"Import multivolume using {channels[pos_informative_channel]} channel..."
        )
        utils.execute_command(exec_config)

        # Merge channels
        self.logger.info("Merging channels step...")
        merge_config = {
            "s": params_multivolume["projout"],
            "d": self.__stitched_folder,
            "sfmt": '"TIFF (unstitched, 3D)"',
            "dfmt": '"TIFF (tiled, 4D)"',
            "cpu_params": config["merge"]["cpu_params"],
            "width": config["merge"]["slice_extent"][0],
            "height": config["merge"]["slice_extent"][1],
            "depth": config["merge"]["slice_extent"][2],
            # 'clist':'0'
        }

        exec_config["command"] = self.merge_multivolume_cmd(merge_config)
        utils.execute_command(exec_config)

    def stitch_single_channels(
        self, config: dict, exec_config: dict, channel: str
    ) -> None:
        """
        Stitch a dataset with a single channel.

        Parameters
        ------------------------
        config: dict
            Configuration dictionary used to process the channels.
        exec_config: dict
            Configuration for command line execution. Mostly for logger.
        channel:str
            Name of the dataset channel that will be imported
        """

        # Step 1
        exec_config["command"] = self.import_step_cmd(
            config["import_data"], channel
        )
        self.logger.info("Import step...")
        utils.execute_command(exec_config)

        # Step 2
        self.logger.info("Align step...")
        exec_config["command"] = self.align_step_cmd(config["align"], channel)
        utils.execute_command(exec_config)

        # Step 3
        self.logger.info("Projection step...")
        exec_config["command"] = self.input_output_step_cmd(
            "displproj",
            f"xml_displcomp_{channel}.xml",
            f"xml_displproj_{channel}.xml",
            channel,
        )
        utils.execute_command(exec_config)

        # Step 4
        self.logger.info("Threshold step...")
        threshold_cnf = {
            "threshold": config["threshold"]["reliability_threshold"]
        }
        exec_config["command"] = self.input_output_step_cmd(
            "displthres",
            f"xml_displproj_{channel}.xml",
            f"xml_displthres_{channel}.xml",
            channel,
            threshold_cnf,
        )
        utils.execute_command(exec_config)

        # Step 5
        self.logger.info("Placing tiles step...")
        exec_config["command"] = self.input_output_step_cmd(
            "placetiles",
            f"xml_displthres_{channel}.xml",
            f"xml_merging_{channel}.xml",
            channel,
        )
        utils.execute_command(exec_config)

        # Step 6
        self.logger.info("Merging step...")
        exec_config["command"] = self.merge_step_cmd(config["merge"], channel)
        utils.execute_command(exec_config)

    def execute_pipeline(self, config: dict, channels: List[str]) -> None:
        """
        Executes the terastitcher's stitching pipeline that includes the following steps:
        Import, Align, Project, Threshold, Place and Merge. Please refer to the following
        link for more information: https://github.com/abria/TeraStitcher/wiki/Stitching-pipeline

        Parameters
        ------------------------
        config: dict
            Configuration dictionary for the stitching pipeline. It should include the configuration
            for each of the steps in the pipeline. i.e. {'import': {...}, 'align': {...}, ...}
        channels:List[str]
            List with channel names that will be processed by pystripe.
            Sigma1 and sigma2 values in the lists belong to each of the channels respectively.
        """

        exec_config = {
            "command": "",
            "verbose": self.__verbose,
            "stdout_log_file": self.stdout_log_file,
            "logger": self.logger,
            # Checking if stdout log file exists
            "exists_stdout": os.path.exists(self.stdout_log_file),
        }

        if self.preprocessing:
            self.__execute_preprocessing_steps(exec_config, channels)

            if "pystripe" in self.preprocessing:
                self.__input_data = self.__preprocessing_folder.joinpath(
                    "destriped"
                )

        if len(channels) > 1:
            self.logger.info(
                f"Processing {channels} channels with informative channel {channels[config['stitch_channel']]}"
            )

            self.stitch_multiple_channels(
                config, exec_config, channels, config["stitch_channel"]
            )

        else:
            self.logger.info(f"Processing single channel {channels[0]}")

            self.stitch_single_channels(config, exec_config, channels[0])

        if config["clean_output"]:
            utils.delete_folder(
                self.__preprocessing_folder.joinpath("destriped"),
                self.__verbose,
            )

        self.logger.info("Converting to OME-Zarr...")
        self.convert_to_ome_zarr(config["ome_zarr_params"], channels)

        ng_config = config["ome_zarr_params"].copy()

        ng_config["ng_base_url"] = config["visualization"]["ng_base_url"]
        ng_config["mount_service"] = config["visualization"]["mount_service"]
        ng_config["bucket_path"] = config["visualization"]["bucket_path"]

        self.create_ng_link(ng_config, channels)

        if config["clean_output"]:
            utils.delete_folder(self.__preprocessing_folder, self.__verbose)


def find_channels(
    path: PathLike, channel_regex: str = r"Ex_([0-9]*)_Em_([0-9]*)$"
):
    """
    Find image channels of a dataset using a regular expression.

    Parameters
    ------------------------
    path:PathLike
        Dataset path
    channel_regex:str
        Regular expression for filtering folders in dataset path.

    Returns
    ------------------------
    List[str]:
        List with the image channels. Empty list if it does not find any channels
        with the given regular expression.
    """
    return [
        path for path in os.listdir(path) if re.search(channel_regex, path)
    ]


def execute_terastitcher(
    input_data: PathLike,
    output_folder: PathLike,
    preprocessed_data: PathLike,
    config_teras: PathLike,
) -> None:

    """
    Executes terastitcher with in-command parameters. It could be on-premise or in the cloud.
    If the process in being carried-out on a GCP VM (i.e. VertexAI jupyter notebook), the
    corresponding buckets will be loaded.

    Parameters
    ------------------------
    input_data: PathLike
        Path where the data is located.

    output_folder: PathLike
        Path where the data will be saved. The module adds the timestamp and '_stitched' suffix.
        e.g. path/to/file/dataset_name -> path/to/file/dataset_name_%Y_%m_%d_%H_%M_%S_stitched

    preprocessed_data: PathLike
        Path where the preprocessed data will be saved (this includes terastitcher output).
        The module adds the timestamp and '_preprocessed' suffix. e.g.
        path/to/file/dataset_name -> path/to/file/dataset_name_%Y_%m_%d_%H_%M_%S_preprocessed

    config_teras: Dict
        Dictionary with terastitcher's configuration.

    """

    # Setting handling error to unmounting cloud for any unexpected error
    def onAnyError(exception_type, value, traceback):

        logger = logging.getLogger(__name__)

        if issubclass(exception_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        else:
            logger.error(
                "Error while executing stitching pipeline: ",
                exc_info=(exception_type, value, traceback),
            )

    sys.excepthook = onAnyError

    # Adding timestamps
    time_stamp = utils.generate_timestamp()
    preprocessed_data = preprocessed_data + "_preprocessed_" + time_stamp
    output_folder = output_folder + "_stitched_" + time_stamp

    regexpression = config_teras["regex_channels"]
    regexpression = "({})".format(regexpression)
    channels = find_channels(input_data, regexpression)
    stitch_channel = config_teras["stitch_channel"]
    len_channels = len(channels)

    if not len_channels or config_teras["stitch_channel"] > len_channels:
        raise ValueError(
            f"Please, check the regular expression for obtaining channels: {channels} and the stitch_channel parameter: {stitch_channel}."
        )

    else:

        try:
            config_teras["preprocessing_steps"]["pystripe"]["input"] = Path(
                input_data
            )
            config_teras["preprocessing_steps"]["pystripe"]["output"] = Path(
                preprocessed_data
            ).joinpath("destriped")
        except KeyError:
            config_teras["preprocessing_steps"] = None

        terastitcher_tool = TeraStitcher(
            input_data=input_data,
            output_folder=output_folder,
            preprocessing_folder=preprocessed_data,
            parallel=True,
            computation="cpu",
            pyscripts_path=config_teras["pyscripts_path"],
            verbose=config_teras["verbose"],
            preprocessing=config_teras["preprocessing_steps"],
        )

        # Saving log command
        terastitcher_cmd = f"$ python terastitcher.py --input_data {input_data} --output_data {output_folder}\n"
        utils.save_string_to_txt(
            terastitcher_cmd, terastitcher_tool.stdout_log_file
        )

        terastitcher_tool.execute_pipeline(config_teras, channels)

    return output_folder


def process_multiple_datasets() -> None:
    default_config = get_default_config()
    # print(default_config)

    mod = ArgSchemaParser(
        input_data=default_config, schema_type=PipelineParams
    )

    args = mod.args

    print("Overwriting datasets...")

    datasets = [
        # {
        #     'input_data': 'Y:/SmartSPIM_Data/2022_07_19/20220719_10_49_05_622386',
        #     'output_data': 'W:/SmartSPIM/ephys/SmartSPIM_622386_2022_07_19_10_49_05_stitched'
        # },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_07_19/20220719_13_57_15_619778",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_619778_2022_07_19_13_57_15_stitched",
        },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_07_19/20220719_17_48_39_595922",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_595922_2022_07_19_17_48_39_stitched",
        },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_08_24/20220824_12_13_29_625749",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_625749_2022_08_24_12_13_29_stitched",
        },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_08_24/20220824_15_33_08_634568",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_634568_2022_08_24_15_33_08_stitched",
        },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_08_24/20220824_19_14_17_634571",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_634571_2022_08_24_19_14_17_stitched",
        },
        {
            "input_data": "Y:/SmartSPIM_Data/2022_08_24/20220824_22_56_15_634569",
            "output_data": "W:/SmartSPIM/ephys/SmartSPIM_634569_2022_08_24_22_56_15_stitched",
        },
    ]

    for dataset in datasets:
        print(f"Processing {dataset['input_data']} dataset")

        execute_terastitcher(
            input_data=dataset["input_data"],
            output_folder=dataset["output_data"],
            config_teras=args,
        )


def main() -> str:
    default_config = get_default_config()

    mod = ArgSchemaParser(
        input_data=default_config, schema_type=PipelineParams
    )

    args = mod.args

    output_folder = execute_terastitcher(
        input_data=args["input_data"],
        output_folder=args["output_data"],
        preprocessed_data=args["preprocessed_data"],
        config_teras=args,
    )

    return output_folder


if __name__ == "__main__":
    main()
    # process_multiple_datasets()
