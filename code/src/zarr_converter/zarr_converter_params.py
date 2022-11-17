from argschema import ArgSchema, ArgSchemaParser
from argschema.fields import Float, InputDir, Int, List, Nested, Str
from argschema.schemas import DefaultSchema


class OmeZarrParams(DefaultSchema):

    codec = Str(
        required=False,
        metadata={"description": "Parameter for ome-zarr compressor"},
        dump_default="zstd",
    )

    clevel = Int(
        required=False,
        metadata={"description": "Parameter for ome-zarr compressor"},
        dump_default=1,
    )

    scale_factor = List(
        Int(),
        required=True,
        metadata={"description": "scale factor for each image axis"},
        cli_as_single_argument=True,
        # dump_default=[2, 2, 2]
    )

    physical_pixels = List(
        Float(),
        required=True,
        metadata={
            "description": "Physical pixel sizes in microns in ZYX order"
        },
        cli_as_single_argument=True,
        dump_default=[2.0, 1.8, 1.8],
    )

    pyramid_levels = Int(
        required=False,
        metadata={
            "description": "number of pyramid levels for ome-zarr multiscale"
        },
        dump_default=5,
    )


class ZarrConvertParams(ArgSchema):
    input_data = InputDir(
        required=True,
        metadata={"description": "Path where the data is located"},
    )

    output_data = Str(
        required=True,
        metadata={"description": "Path where the data will be saved"},
    )

    writer = Nested(OmeZarrParams, required=False)


def get_default_config() -> dict:
    return {
        "writer": {
            "codec": "zstd",
            "clevel": 1,
            "scale_factor": [2, 2, 2],
            "pyramid_levels": 5,
        }
    }
