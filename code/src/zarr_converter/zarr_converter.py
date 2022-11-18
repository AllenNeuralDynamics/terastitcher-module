import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask
import numpy as np
import pims
import xarray_multiscale
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeZarrWriter
from argschema import ArgSchemaParser
from dask.array import concatenate, pad, zeros
from dask.array.core import Array
from dask.base import tokenize
from dask.distributed import Client, LocalCluster, performance_report, progress
from natsort import natsorted
from numcodecs import blosc
from skimage.io import imread as sk_imread

from .zarr_converter_params import ZarrConvertParams, get_default_config

PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
blosc.use_threads = False

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

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.

    dim: int
        Number of dimensions that the array will be padded
    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def add_leading_dim(data):
    return data[None, ...]


def read_image_directory_structure(folder_dir: PathLike) -> dict:
    """
    Creates a dictionary representation of all the images saved by folder/col_N/row_N/images_N.[file_extention]

    folder_dir:PathLike
        Path to the folder where the images are stored

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {
            channel_1: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
            .
            .
            .
            channel_n: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
        }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = [
        folder_dir.joinpath(folder)
        for folder in os.listdir(folder_dir)
        if os.path.isdir(folder_dir.joinpath(folder))
    ]

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}

        cols = natsorted(os.listdir(channel_paths[channel_idx]))

        for col in cols:
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}

                rows = natsorted(os.listdir(possible_col))

                for row in rows:
                    possible_row = (
                        channel_paths[channel_idx].joinpath(col).joinpath(row)
                    )

                    if os.path.isdir(possible_row):
                        directory_structure[channel_paths[channel_idx]][col][
                            row
                        ] = natsorted(os.listdir(possible_row))
    return directory_structure


def lazy_tiff_reader(filename: PathLike):  # , chunksize:tuple, dtype):
    """
    Creates a dask array to read an image located in a specific path.

    filename:PathLike
        Path to the image

    chunksize:
        Image chunksize

    dtype:
        Image dtype

    Returns
    ------------------------
    dask.array.core.Array
        Array representing the image data
    """
    name = "imread-%s" % tokenize(filename, map(os.path.getmtime, filename))

    with pims.open(filename) as imgs:
        shape = (1,) + (len(imgs),) + imgs.frame_shape
        dtype = np.dtype(imgs.pixel_type)

    key = [(name,) + (0,) * len(shape)]
    value = [(add_leading_dim, (sk_imread, filename))]
    dask_arr = dict(zip(key, value))
    chunks = tuple((d,) for d in shape)

    return Array(dask_arr, name, chunks, dtype)


def fix_image_diff_dims(new_arr, chunksize, len_chunks, work_axis):
    zeros_dim = []
    diff_dim = -1
    c = 0

    for chunk_idx in range(len_chunks):
        new_chunk_dim = new_arr.chunksize[chunk_idx]

        if new_chunk_dim != chunksize[chunk_idx]:
            c += 1
            diff_dim = chunk_idx

        zeros_dim.append(abs(chunksize[chunk_idx] - new_chunk_dim))

    if c > 1:
        raise ValueError("Block has two different dimensions")
    else:

        if (diff_dim - len_chunks) == work_axis:
            return new_arr

        n_pad = tuple(tuple((0, dim)) for dim in zeros_dim)
        new_arr = pad(
            new_arr, pad_width=n_pad, mode="constant", constant_values=0
        ).rechunk(chunksize)

    return new_arr


def concatenate_dask_arrays(arr_1, arr_2, axis):

    shape_arr_1 = arr_1.shape
    shape_arr_2 = arr_2.shape

    if shape_arr_1 != shape_arr_2:
        # Adjust arr_size

        slices = []
        dims = len(shape_arr_1)
        for shape_dim_idx in range(dims):

            if shape_arr_1[shape_dim_idx] > shape_arr_2[shape_dim_idx] and (
                shape_dim_idx - dims != axis
            ):
                raise ValueError(
                    f"Array 1 {shape_arr_1} must have a smaller shape than array 2 {shape_arr_2} except for the axis dimension {shape_dim_idx} {dims} {shape_dim_idx - dims} {axis}"
                )

            if shape_arr_1[shape_dim_idx] != shape_arr_2[shape_dim_idx]:
                slices.append(slice(0, shape_arr_1[shape_dim_idx]))

            else:
                slices.append(slice(None))

        slices = tuple(slices)
        arr_2 = arr_2[slices]

    try:
        res = concatenate([arr_1, arr_2], axis=axis)
    except:
        raise ValueError(
            f"Unable to cancat arrays - Shape 1: {shape_arr_1} shape 2: {shape_arr_2}"
        )

    return res


def read_chunked_stitched_image_per_channel(
    directory_structure: dict,
    channel_name: str,
    start_slice: int,
    end_slice: int,
    chunksize: tuple,
    dtype,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume based on image chunks preserving the chunksize.

    directory_structure:dict
        dictionary to store paths of images with the following structure:
        {
            channel_1: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
            .
            .
            .
            channel_n: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
        }

    channel_name : str
        Channel name to reconstruct the image volume
    Returns
    ------------------------
    ArrayLike
        Array with the image volume

    """
    concat_z_3d_blocks = concat_horizontals = horizontal = None

    # Getting col structure
    cols = list(directory_structure.values())[0]
    cols_paths = list(cols.keys())
    # process = multiprocessing.current_process()
    first = True
    len_chunks = len(chunksize)

    for slice_pos in range(start_slice, end_slice):
        idx_col = 0
        idx_row = 0

        concat_horizontals = None

        for column_name in cols_paths:
            idx_row = 0
            horizontal = []

            for row_name in directory_structure[channel_name][column_name]:
                # col_row_images = len(directory_structure[channel_name][column_name][row_name])

                try:
                    # if col_row_images > slice_pos:
                    slice_name = directory_structure[channel_name][
                        column_name
                    ][row_name][slice_pos]
                    filepath = str(
                        channel_name.joinpath(column_name)
                        .joinpath(row_name)
                        .joinpath(slice_name)
                    )

                    new_arr = lazy_tiff_reader(filepath)

                    if new_arr.chunksize != chunksize:

                        # Fix chunksize when it's not on the axis concat
                        new_arr = fix_image_diff_dims(
                            new_arr, chunksize, len_chunks, -1, dtype
                        )

                    horizontal.append(new_arr)

                except:  # else:
                    # Due to the different brain volume and output shape, sometimes the stitching does not include last horizontal
                    # To match array shape, we include a black background since this mostly happens at the end of the images
                    # print(
                    #     f"Not able to read read horizontal. Array shape: {slice_pos} {directory_structure[channel_name][column_name][row_name]}: {len(directory_structure[channel_name][column_name][row_name])} {len(horizontal)} \n"
                    # )
                    horizontal.append(zeros(chunksize, dtype))

                idx_row += 1

            # Concatenating horizontally lazy images
            horizontal = concatenate(horizontal, axis=-1)

            if not idx_col:
                concat_horizontals = horizontal
            else:
                concat_horizontals = concatenate_dask_arrays(
                    arr_1=concat_horizontals, arr_2=horizontal, axis=-2
                )

            idx_col += 1

        if first:
            concat_z_3d_blocks = concat_horizontals
            first = False

        else:
            concat_z_3d_blocks = concatenate_dask_arrays(
                arr_1=concat_z_3d_blocks, arr_2=concat_horizontals, axis=-3
            )

    return concat_z_3d_blocks, [start_slice, end_slice]


def _read_chunked_stitched_image_per_channel(args_dict: dict):
    return read_chunked_stitched_image_per_channel(**args_dict)


def channel_parallel_reading(
    directory_structure: dict,
    channel_idx: int,
    sample_img: dask.array.core.Array,
    workers: Optional[int] = 0,
    chunks: Optional[int] = 1,
) -> dask.array.core.Array:

    if workers == 0:
        workers = multiprocessing.cpu_count()

    cols = list(directory_structure.values())[0]
    n_images = len(list(list(cols.values())[0].values())[0])
    LOGGER.info(f"n_images: {n_images}")

    channel_paths = list(directory_structure.keys())
    dask_array = None
    ensure_parallel = True

    if n_images < workers and ensure_parallel:
        workers = n_images

    if n_images < workers:
        dask_array = read_chunked_stitched_image_per_channel(
            directory_structure=directory_structure,
            channel_name=channel_paths[channel_idx],
            start_slice=0,
            end_slice=n_images,
            chunksize=sample_img.chunksize,
            dtype=sample_img.dtype,
        )[0]
        LOGGER.info(f"No need for parallel reading... {dask_array}")

    else:
        images_per_worker = n_images // workers
        LOGGER.info(
            f"Setting workers to {workers} - {images_per_worker} - total images: {n_images}"
        )

        # Getting 5 dim image TCZYX
        args = []
        start_slice = 0
        end_slice = images_per_worker

        for idx_worker in range(workers):
            arg_dict = {
                "directory_structure": directory_structure,
                "channel_name": channel_paths[channel_idx],
                "start_slice": start_slice,
                "end_slice": end_slice,
                "chunksize": sample_img.chunksize,
                "dtype": sample_img.dtype,
            }

            args.append(arg_dict)

            if idx_worker + 1 == workers - 1:
                start_slice = end_slice
                end_slice = n_images
            else:
                start_slice = end_slice
                end_slice += images_per_worker

        res = []
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap(
                _read_chunked_stitched_image_per_channel,
                args,
                chunksize=chunks,
            )

            for pos in results:
                res.append(pos)

        for res_idx in range(len(res)):
            if not res_idx:
                dask_array = res[res_idx][0]
            else:
                dask_array = concatenate(
                    [dask_array, res[res_idx][0]], axis=-3
                )

            LOGGER.info(f"Slides: {res[res_idx][1]}")

    return dask_array


def parallel_read_chunked_stitched_multichannel_image(
    directory_structure: dict,
    sample_img: dask.array.core.Array,
    workers: Optional[int] = 0,
) -> ArrayLike:
    """
    Creates a dask array of the whole image volume based on image chunks preserving the chunksize.

    directory_structure:dict
        dictionary to store paths of images with the following structure:
        {
            channel_1: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
            .
            .
            .
            channel_n: {
                col_1: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
                .
                .
                .
                col_n: {row_1: [image_1, image_2, ..., image_n] ... row_1: [image_1, image_2, ..., image_n]}
            }
        }

    sample_img : dask.array.core.Array
        Sample image of the dataset
    Returns
    ------------------------
    ArrayLike
        Array with the image volume

    """

    multichannel_image = None

    channel_paths = list(directory_structure.keys())

    multichannels = []
    LOGGER.info(f"Channel in directory structure: {channel_paths}")
    
    for channel_idx in range(len(channel_paths)):
        LOGGER.info(f"Reading images from {channel_paths[channel_idx]}")
        start_time = time.time()
        read_chunked_channel = channel_parallel_reading(
            directory_structure, channel_idx, sample_img, workers=workers
        )
        end_time = time.time()

        LOGGER.info(
            f"Time reading single channel image: {end_time - start_time}"
        )

        # Padding to 4D if necessary
        read_chunked_channel = pad_array_n_d(read_chunked_channel, 4)
        multichannels.append(read_chunked_channel)

    if len(multichannels) > 1:
        multichannel_image = concatenate(multichannels, axis=0)
    else:
        multichannel_image = multichannels[0]

    return multichannel_image


class ZarrConverter:
    def __init__(
        self,
        input_data: PathLike,
        output_data: PathLike,
        blosc_config: dict,
        channels: List[str] = None,
        physical_pixels: List[float] = None,
    ) -> None:

        self.input_data = input_data
        self.output_data = output_data
        self.physical_pixels = None
        self.dask_folder = Path("/root/capsule/scratch")

        if physical_pixels:
            self.physical_pixels = PhysicalPixelSizes(
                physical_pixels[0], physical_pixels[1], physical_pixels[2]
            )

        self.writer = OmeZarrWriter(output_data)

        self.opts = {
            "compressor": blosc.Blosc(
                cname=blosc_config["codec"],
                clevel=blosc_config["clevel"],
                shuffle=blosc.SHUFFLE,
            )
        }

        self.channels = channels
        self.channel_colors = None

        if channels != None:
            colors = [
                0xFF0000,  # Red
                0x00FF00,  # green
                0xFF00FF,  # Purple
                0xFFFF00,  # Yellow
            ]
            self.channel_colors = colors[: len(self.channels)]

        # get_blosc_codec(writer_config['codec'], writer_config['clevel'])

    def compute_pyramid(
        self, data: dask.array.core.Array, n_lvls: int, scale_axis: Tuple[int]
    ) -> List[dask.array.core.Array]:

        """
        Computes the pyramid levels given an input full resolution image data

        Parameters
        ------------------------
        data: dask.array.core.Array
            Dask array of the image data

        n_lvls: int
            Number of downsampling levels that will be applied to the original image

        scale_axis: Tuple[int]
            Scaling applied to each axis

        Returns
        ------------------------
        List[dask.array.core.Array]:
            List with the downsampled image(s)
        """

        pyramid = xarray_multiscale.multiscale(
            data,
            xarray_multiscale.reducers.windowed_mean,  # func
            scale_axis,  # scale factors
            depth=n_lvls - 1,
            preserve_dtype=True,
        )

        return [arr.data for arr in pyramid]

    def get_pyramid_metadata(self) -> dict:
        """
        Gets pyramid metadata in OMEZarr format

        Returns
        ------------------------
        dict:
            Dictionary with the downscaling OMEZarr metadata
        """

        return {
            "metadata": {
                "description": "Downscaling implementation based on the windowed mean of the original array",
                "method": "xarray_multiscale.reducers.windowed_mean",
                "version": str(xarray_multiscale.__version__),
                "args": "[false]",
                "kwargs": {},  # No extra parameters were used different from the orig. array and scales
            }
        }

    def convert(
        self, writer_config: dict, image_name: str = "zarr_multiscale.zarr"
    ) -> None:

        """
        Executes the OME-Zarr conversion

        Parameters
        ------------------------

        writer_config: dict
            OME-Zarr writer configuration

        image_name: str
        Name of the image

        """

        sample_img = None
        directory_structure = read_image_directory_structure(self.input_data)
        for channel_dir, val in directory_structure.items():
            # n_cols = len(val.keys())
            for col_name, rows in val.items():
                # n_rows = len(rows.keys())

                for row_name, images in rows.items():
                    # print(f"Channel: {str(channel_dir).split('/')[-1]} colname: {col_name} rowname: {row_name} cols_rows: {n_cols} {n_rows} images: {len(images)}")

                    if not isinstance(sample_img, dask.array.core.Array):
                        sample_path = (
                            channel_dir.joinpath(col_name)
                            .joinpath(row_name)
                            .joinpath(images[0])
                        )
                        sample_img = lazy_tiff_reader(str(sample_path))
                        # sample_img = imread(str(sample_path))

        LOGGER.info(f"Sample image in dataset: {sample_img} Channels: {directory_structure.keys()}")

        workers = 0
        image = pad_array_n_d(
            parallel_read_chunked_stitched_multichannel_image(
                directory_structure, sample_img, workers
            )
        )

        LOGGER.info(f"Multichannel image: {image} {image.npartitions}")

        if not isinstance(image, dask.array.core.Array):
            raise ValueError(
                f"There was an error reading the images from: {self.input_data}"
            )

        scale_axis = []
        for axis in range(
            len(image.shape) - len(writer_config["scale_factor"])
        ):
            scale_axis.append(1)

        scale_axis.extend(list(writer_config["scale_factor"]))
        scale_axis = tuple(scale_axis)

        start_time = time.time()

        dask.config.set(
            {
                "temporary-directory": self.dask_folder,
                "local_directory": self.dask_folder,
                "tcp-timeout": "60s",
                "array.chunk-size": "384MiB",
                "distributed.comm.timeouts": {"connect": "60s", "tcp": "60s"},
                "distributed.scheduler.bandwidth": 100000000,
                "distributed.worker.memory.rebalance.measure": "managed_in_memory",
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": False,
                "distributed.worker.memory.pause": False,
                "distributed.worker.memory.terminate": False
                # 'distributed.scheduler.unknown-task-duration': '15m',
                # 'distributed.scheduler.default-task-durations': '2h',
            }
        )

        cluster = LocalCluster()
        client = Client(cluster)

        with performance_report(filename="dask-report.html"):

            pyramid_data = self.compute_pyramid(
                image, writer_config["pyramid_levels"], scale_axis
            )

            pyramid_data = [pad_array_n_d(pyramid) for pyramid in pyramid_data]
            channel_names = self.channels if self.channels else None
            channel_colors = (
                self.channel_colors if self.channel_colors else None
            )

            dask_jobs = self.writer.write_multiscale(
                pyramid=pyramid_data,  # must be 5D TCZYX
                image_name=image_name,
                physical_pixel_sizes=self.physical_pixels,
                channel_names=channel_names,
                channel_colors=channel_colors,
                scale_factor=writer_config["scale_factor"],
                chunks=pyramid_data[0].chunksize,
                storage_options=self.opts,
                compute_dask=True,
                # **self.get_pyramid_metadata()
            )

            if len(dask_jobs):
                dask_jobs = dask.persist(*dask_jobs, compute_chunk_sizes=True)
                progress(dask_jobs)

        client.close()

        end_time = time.time()
        LOGGER.info(f"Done converting dataset. Took {end_time - start_time}s.")


def main():
    default_config = get_default_config()

    mod = ArgSchemaParser(
        input_data=default_config, schema_type=ZarrConvertParams
    )

    args = mod.args

    zarr_converter = ZarrConverter(
        input_data=args["input_data"],
        output_data=args["output_data"],
        blosc_config={
            "codec": args["writer"]["codec"],
            "clevel": args["writer"]["clevel"],
        },
        channels=["Ex_488_Em_525", "Ex_561_Em_600"],
        physical_pixels=[2.0, 1.8, 1.8],
    )

    zarr_converter.convert(args["writer"], "zarr_test.zarr")


if __name__ == "__main__":
    main()
