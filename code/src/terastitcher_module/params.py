import os
import platform
import pprint as pp
from pathlib import Path

import yaml
from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import (Boolean, Float, InputDir, InputFile, Int, List,
                              Nested, NumpyArray, Str)
from argschema.schemas import DefaultSchema
from marshmallow import validate
from zarr_converter import OmeZarrParams


class InputFileBasedLinux(InputFile):
    """

    InputFileBasedOS is a :class:`argschema.fields.InputFile` subclass which is a path to
    a file location which can be read by the user depending if it's on Linux or not.

    """

    def _validate(self, value):
        if platform.system() != "Windows":
            super()._validate(value)


class InputDirGCloud(InputDir):
    """

    InputDirGCloud is a :class:`argschema.fields.InputDir` subclass which is a path to
    a directory location which will be validated if the user is not on GCloud.

    """

    def _validate(self, value):
        if not value.startswith("gs://"):
            super()._validate(value)
        else:
            # Validate after mounting
            pass


class ImportParameters(DefaultSchema):

    ref1 = Str(
        required=False,
        metadata={"description": "First axis of the dataset reference system"},
        dump_default="X",
    )

    ref2 = Str(
        required=False,
        metadata={
            "description": "Second axis of the dataset reference system"
        },
        dump_default="Y",
    )

    ref3 = Str(
        required=False,
        metadata={"description": "Third axis of the dataset reference system"},
        dump_default="D",
    )

    vxl1 = Float(
        required=False,
        matadata={"description": "Voxel size along first axis in microns"},
        dump_default=1.8,
    )

    vxl2 = Float(
        required=False,
        matadata={"description": "Voxel size along second axis in microns"},
        dump_default=1.8,
    )

    vxl3 = Float(
        required=False,
        matadata={"description": "Voxel size along third axis in microns"},
        dump_default=2,
    )

    additional_params = List(
        Str(), required=False, cli_as_single_argument=True
    )


class CPUParams(DefaultSchema):

    estimate_processes = Boolean(
        required=False,
        matadata={
            "description": "Estimate processes option for align or merge steps"
        },
        dump_default=False,
    )

    image_depth = Int(
        required=False,
        metadata={
            "description": "Layer thickness along Z axis when processing layer by layer"
        },
    )

    number_processes = Int(
        required=False,
        metadata={"description": "Degree of parallelization in gpu or cpu"},
    )

    hostfile = InputFileBasedLinux(
        required=True,
        metadata={
            "description": "Path to MPI hostfile. Only for Linux kernel machines"
        },
        dump_default="/home/hostfile",
    )

    additional_params = List(
        Str(),
        required=False,
        dump_default=["use-hwthread-cpus", "allow-run-as-root"],
        cli_as_single_argument=True,
    )


class AlignParameters(DefaultSchema):

    subvoldim = Int(
        required=False,
        metadata={
            "description": "Layer thickness along Z axis when processing layer by layer"
        },
        dump_default=100,
    )

    cpu_params = Nested(CPUParams)


class ThresholdParameters(DefaultSchema):

    reliability_threshold = Float(
        required=False,
        metadata={
            "description": "Reliability threshold applied to the computed displacements to select the most reliable ones"
        },
        dump_default=0.7,
        validate=validate.Range(
            min=0, min_inclusive=False, max=1, max_inclusive=True
        ),
    )


class MergeParameters(DefaultSchema):

    slice_extent = List(
        Int(),
        required=True,
        metadata={
            "description": "Supposing the output image is saved in a tiled format, this is an array that contains the slice size of output tiles in order [slicewidth, sliceheight, slicedepth]"
        },
        cli_as_single_argument=True,
        dump_default=[20000, 20000, 0],
    )

    volout_plugin = Str(
        required=False,
        metadata={
            "description": "Tiling images output. For 2D 'TiledXY|2Dseries', for 3D 'TiledXY|3Dseries'"
        },
        dump_default='"TiledXY|2Dseries"',
    )

    cpu_params = Nested(CPUParams)


class PystripeParams(DefaultSchema):
    # input and output are already defined in PipelineParams Class
    sigma1 = List(
        Int(),
        required=False,
        metadata={
            "description": "bandwidth of the stripe filter for the foreground for each channel"
        },
        cli_as_single_argument=True,
        dump_default=[256, 800, 800],
    )

    sigma2 = List(
        Int(),
        required=False,
        metadata={
            "description": "bandwidth of the stripe filter for the background for each channel"
        },
        cli_as_single_argument=True,
        dump_default=[256, 800, 800],
    )

    workers = Int(
        required=False,
        metadata={
            "description": "number of cpu workers to use in batch processing"
        },
        dump_default=8,
    )

    output_format = Str(
        required=False,
        metadata={
            "description": "Output format for the images in pystripe step"
        },
        dump_default=".tiff",
    )


class PreprocessingSteps(DefaultSchema):
    pystripe = Nested(PystripeParams, required=False)


class Visualization(DefaultSchema):
    ng_base_url = Str(
        required=True,
        metadata={"description": "Base url for neuroglancer web app"},
    )

    mount_service = Str(
        required=True,
        metadata={
            "description": "Set to s3 if the dataset will be saved in a Amazon Bucket, gs for a Google Bucket"
        },
        dump_default="s3",
    )

    bucket_path = Str(
        required=True,
        metadata={"description": "Amazon Bucket or Google Bucket name"},
    )


class PipelineParams(ArgSchema):

    input_data = InputDirGCloud(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    output_data = Str(
        required=True,
        metadata={"description": "Path where the data will be saved"},
    )

    preprocessed_data = Str(
        required=True,
        metadata={
            "description": "Path where the preprocessed data will be saved (this includes terastitcher output)"
        },
    )

    stitch_channel = Int(
        required=True,
        metadata={
            "description": "Position of the informative channel for stitching"
        },
        dump_default=0,
        validate=validate.Range(
            min=0, min_inclusive=True, max=3, max_inclusive=True
        ),
    )

    regex_channels = Str(
        required=False,
        metadata={"description": "Path where the data will be saved"},
        dump_default="Ex_([0-9]*)_Em_([0-9]*)$",
    )

    pyscripts_path = InputDir(
        required=True,
        metadata={
            "description": "Path to stitched parallel scripts (parastitcher and paraconverter must be there)."
        },
    )

    # Processing params
    preprocessing_steps = Nested(PreprocessingSteps, required=False)
    import_data = Nested(ImportParameters, required=True)
    align = Nested(AlignParameters, required=False)
    threshold = Nested(ThresholdParameters, required=False)
    merge = Nested(MergeParameters, required=False)
    verbose = Boolean(
        required=False,
        matadata={"description": "Set verbose for stitching."},
        dump_default=True,
    )

    # Conversion params
    ome_zarr_params = Nested(OmeZarrParams, required=False)

    clean_output = Boolean(
        required=False,
        matadata={
            "description": "Set True if you want to delete intermediate output images (e.g. pystripe, terastitcher) and keep only OME-Zarr images. Set False otherwise."
        },
        dump_default=False,
    )

    visualization = Nested(Visualization, required=True)


def get_default_config(filename: str = "default_config.yaml"):

    filename = Path(os.path.dirname(__file__)).joinpath(filename)

    config = None
    try:
        with open(filename, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config


if __name__ == "__main__":
    # this defines a default dictionary that will be used if input_json is not specified
    example = get_default_config()
    mod = ArgSchemaParser(input_data=example, schema_type=PipelineParams)

    pp.pprint(mod.args)
