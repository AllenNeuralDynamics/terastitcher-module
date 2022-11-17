import os
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask
import numpy as np
import tifffile
import zarr
from parameterized import parameterized
from tests import params
from utils import utils
from zarr_converter import ZarrConverter

PathLike = Union[str, Path]


def _create_tiffs_files(
    path_to_tiffs: str, n_tiffs: int = 4, shape: Tuple[int] = (256, 256, 256)
):
    for idx_tiffs in range(n_tiffs):
        tiff_data = np.ones(shape, dtype=np.uint16)
        tifffile.imwrite(
            os.path.join(path_to_tiffs, f"data_{idx_tiffs}.tif"), tiff_data
        )


def _create_multichannel_tiff_files(
    root: str,
    n_tiffs: int = 4,
    n_channels: int = 3,
    shape: Tuple[int] = (256, 256, 256),
):

    for channel_idx in range(n_channels):
        channel_path = Path(root).joinpath(f"channel_{channel_idx}/x/y/")
        utils.create_folder(channel_path)
        _create_tiffs_files(channel_path, n_tiffs=n_tiffs, shape=shape)


def _create_zarr_file(
    path_to_files: PathLike,
    output_path: PathLike,
    image_name: Optional[str] = "test_zarr_1.zarr",
    channels: Optional[List[str]] = None,
):
    test_writer = ZarrConverter(
        path_to_files, output_path, {"codec": "zstd", "clevel": 1}, channels
    )

    config = {
        "codec": "zstd",
        "clevel": 1,
        "scale_factor": [2, 2, 2],
        "pyramid_levels": 5,
    }

    test_writer.convert(config, image_name=image_name)


class TestZarrConverter(unittest.TestCase):
    def setUp(self):
        # io folders
        self.maxDiff = None
        self._tmp_dir = tempfile.TemporaryDirectory()
        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_1"
        utils.create_folder(path_to_tiffs)
        _create_tiffs_files(path_to_tiffs)

        # Using stitching hierarchical structure for reading multichannel
        utils.create_folder(f"{self._tmp_dir.name}/tiffs_single_channel_2/x/y")
        _create_tiffs_files(f"{self._tmp_dir.name}/tiffs_single_channel_2/x/y")

        utils.create_folder(f"{self._tmp_dir.name}/converted")

        # Creating multichannel data with hierarchical structure
        utils.create_folder(f"{self._tmp_dir.name}/multichannel")
        _create_multichannel_tiff_files(f"{self._tmp_dir.name}/multichannel")

    def test_convert_to_ome_zarr(self):
        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_2"
        converted_zarr = f"{self._tmp_dir.name}/converted"

        zarr_writer = ZarrConverter(
            path_to_tiffs, converted_zarr, {"codec": "zstd", "clevel": 1}
        )

        config = {
            "codec": "zstd",
            "clevel": 1,
            "scale_factor": [2, 2, 2],
            "pyramid_levels": 3,
        }

        zarr_writer.convert(config, image_name="test_tiffs.zarr")
        self.assertTrue(
            os.path.isdir(f"{self._tmp_dir.name}/converted/test_tiffs.zarr")
        )

    def test_pad_array_n_d(self):
        # io folders
        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_1"

        zarr_writer = ZarrConverter(
            path_to_tiffs, path_to_tiffs, {"codec": "zstd", "clevel": 1}
        )

        # Reading images and padding
        array = zarr_writer.read_channel_image(path_to_tiffs)
        pad_array = zarr_writer.pad_array_n_d(array, 5)
        re_pad_array = zarr_writer.pad_array_n_d(array, 5)

        # Tests
        self.assertEqual(pad_array.ndim, 5)
        with self.assertRaises(ValueError):
            zarr_writer.pad_array_n_d(array, 6)

        self.assertEqual(re_pad_array.ndim, 5)

    @parameterized.expand(params.get_compute_pyramid_params())
    def test_compute_pyramid(self, n_lvls: int, scale_axis: int):
        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_1"
        zarr_writer = ZarrConverter(
            path_to_tiffs, path_to_tiffs, {"codec": "zstd", "clevel": 1}
        )
        array = zarr_writer.read_channel_image(path_to_tiffs)

        array_pyramid = zarr_writer.compute_pyramid(array, n_lvls, scale_axis)
        self.assertIsInstance(array_pyramid, list)
        self.assertEqual(len(array_pyramid), n_lvls)

        for multiscale in array_pyramid:
            self.assertIsInstance(multiscale, dask.array.core.Array)
            self.assertEqual(multiscale.ndim, 4)

    @parameterized.expand(params.get_compute_pyramid_raises_params())
    def test_compute_pyramid_raises(self, n_lvls: int, scale_axis: int):
        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_1"
        zarr_writer = ZarrConverter(
            path_to_tiffs, path_to_tiffs, {"codec": "zstd", "clevel": 1}
        )
        array = zarr_writer.read_channel_image(path_to_tiffs)

        with self.assertRaises(ValueError):
            array_pyramid = zarr_writer.compute_pyramid(
                array, n_lvls, scale_axis
            )

    def _check_single_channel_omero(
        self, omero_metadata: dict, filename: PathLike
    ):
        expected_omero = {
            "channels": [
                {
                    "active": True,
                    "coefficient": 1,
                    "color": "000000",
                    "family": "linear",
                    "inverted": False,
                    "label": f"Channel:{filename}:0",
                    "window": {
                        "end": 1.0,
                        "max": 1.0,
                        "min": 0.0,
                        "start": 0.0,
                    },
                }
            ],
            "id": 1,
            "name": filename,
            "rdefs": {"defaultT": 0, "defaultZ": 128, "model": "color"},
            "version": "0.4",
        }

        self.assertDictEqual(omero_metadata, expected_omero)

    def _check_multiple_channel_omero(
        self, omero_metadata: dict, filename: str, channels: List[str]
    ):
        channels_metadata = []

        for channel_idx in range(len(channels)):
            channels_metadata.append(
                {
                    "active": True,
                    "coefficient": 1,
                    "color": f"00000{channel_idx}",
                    "family": "linear",
                    "inverted": False,
                    "label": channels[channel_idx],
                    "window": {
                        "end": 1.0,
                        "max": 1.0,
                        "min": 0.0,
                        "start": 0.0,
                    },
                },
            )

        expected_omero = {
            "channels": channels_metadata,
            "id": 1,
            "name": filename,
            "rdefs": {"defaultT": 0, "defaultZ": 128, "model": "color"},
            "version": "0.4",
        }
        self.assertDictEqual(omero_metadata, expected_omero)

    def _check_downsamplig_metadata(self, downsampling_metadata: dict):
        expected_downsampling = {
            "args": "[false]",
            "description": "Downscaling implementation based on the windowed mean of the original array",
            "kwargs": {},
            "method": "xarray_multiscale.reducers.windowed_mean",
            "version": "0.2.2",
        }
        self.assertDictEqual(downsampling_metadata, expected_downsampling)

    def _check_single_image_axes(self, axes_metadata: list):
        expected_axes = [
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        self.assertIsInstance(axes_metadata, list)

        for axes_idx in range(len(expected_axes)):
            self.assertDictEqual(
                axes_metadata[axes_idx], expected_axes[axes_idx]
            )

    def _generate_single_channel_dataset_metadata(self, n_scaling=5):

        datasets = []
        x_voxelsize = y_voxelsize = z_voxelsize = 1.0

        for n_scale in range(n_scaling):
            datasets.append(
                {
                    "coordinateTransformations": [
                        {
                            "scale": [
                                1.0,
                                1.0,
                                z_voxelsize,
                                z_voxelsize,
                                z_voxelsize,
                            ],
                            "type": "scale",
                        }
                    ],
                    "path": str(n_scale),
                }
            )

            x_voxelsize += x_voxelsize
            y_voxelsize += y_voxelsize
            z_voxelsize += z_voxelsize

        return datasets

    def _check_single_image_datasets(self, datasets_metadata: list):
        expected_datasets = self._generate_single_channel_dataset_metadata()

        self.assertIsInstance(datasets_metadata, list)
        for dataset_idx in range(len(datasets_metadata)):
            self.assertDictEqual(
                datasets_metadata[dataset_idx], expected_datasets[dataset_idx]
            )

    def test_check_single_channel_zarr_attributes(self):

        path_to_tiffs = f"{self._tmp_dir.name}/tiffs_single_channel_2"
        converted_zarr = f"{self._tmp_dir.name}/converted"
        filename = "test_tiffs_single_channel_2.zarr"

        _create_zarr_file(path_to_tiffs, converted_zarr, filename)

        zarr_file = zarr.open(
            f"{self._tmp_dir.name}/converted/{filename}", mode="r"
        )
        attributes = zarr_file.attrs

        # Checking axes
        self._check_single_image_axes(attributes["multiscales"][0]["axes"])

        # Checking downsampling metadata using xarray_multiscale package
        self._check_downsamplig_metadata(
            attributes["multiscales"][0]["metadata"]
        )

        # Checking datasets metadata
        self._check_single_image_datasets(
            attributes["multiscales"][0]["datasets"]
        )

        # Checking omero metadata
        self._check_single_channel_omero(attributes["omero"], filename)

    def test_check_multichannel_zarr_attributes(self):

        path_to_tiffs = f"{self._tmp_dir.name}/multichannel"
        converted_zarr = f"{self._tmp_dir.name}/converted"
        filename = "test_multichannel.zarr"

        _create_zarr_file(path_to_tiffs, converted_zarr, filename)
        zarr_file = zarr.open(
            f"{self._tmp_dir.name}/converted/{filename}", mode="r"
        )
        attributes = zarr_file.attrs

        # Checking axes
        self._check_single_image_axes(attributes["multiscales"][0]["axes"])

        # Checking downsampling metadata using xarray_multiscale package
        self._check_downsamplig_metadata(
            attributes["multiscales"][0]["metadata"]
        )

        # Checking datasets metadata
        # TODO Include when we fix chunksizes in zarr file
        # self._check_single_image_datasets(attributes['multiscales'][0]['datasets'])

        # Checking omero metadata - channels are created if channels param is not given in converter
        # Channel:{filename}:counter
        # In setup I created a 3 channel dataset
        channels = [
            "Channel:test_multichannel.zarr:0",
            "Channel:test_multichannel.zarr:1",
            "Channel:test_multichannel.zarr:2",
            "Channel:test_multichannel.zarr:3",
        ]
        self._check_multiple_channel_omero(
            attributes["omero"], filename, channels
        )

    def tearDown(self):
        self._tmp_dir.cleanup()
