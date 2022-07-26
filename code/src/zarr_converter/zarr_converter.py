import os
from pathlib import Path
import logging
from dask.distributed import Client, LocalCluster, progress, performance_report
from numcodecs import blosc
import dask
from dask.array.image import imread
from dask.array import moveaxis, concatenate, stack
import argparse
import time
from typing import List, Optional, Union, Tuple, Any
import xarray_multiscale
import numpy as np
from .zarr_converter_params import ZarrConvertParams, get_default_config
from argschema import ArgSchemaParser
from glob import glob
from aicsimageio.types import PhysicalPixelSizes
from xarray import DataArray
from aicsimageio.writers import OmeZarrWriter
from aicsimageio.readers.tiff_reader import TiffReader
from utils import utils
import re

PathLike = Union[str, Path]
ArrayLike = Union[dask.array.core.Array, np.ndarray]
blosc.use_threads = False

def search_images_in_folder(path:PathLike, format_ext_regex:str) -> str:

    level = 0

    # Search until images are found
    for root, dirs, files in os.walk(path):
        if not len(dirs) and len(files):
            if re.match(format_ext_regex, files[0]):
                break
        level += 1
        
    return level

class ZarrConverter():
    
    def __init__(
        self, 
        input_data:PathLike, 
        output_data:PathLike, 
        blosc_config:dict,
        channels:List[str]=None,
        physical_pixels:List[float]=None,
        file_format:List[str]='tif'
    ) -> None:
        
        self.input_data = input_data
        self.output_data = output_data
        self.physical_pixels = None
        self.dask_folder = Path('/root/capsule/scratch')
        
        if physical_pixels:
            self.physical_pixels = PhysicalPixelSizes(physical_pixels[0], physical_pixels[1], physical_pixels[2])
            
        self.file_format = file_format
        
        self.writer = OmeZarrWriter(
            output_data
        )
        
        self.opts = {
            'compressor': blosc.Blosc(cname=blosc_config['codec'], clevel=blosc_config['clevel'], shuffle=blosc.SHUFFLE)
        }
        
        self.channels = channels
        self.channel_colors = None
        
        if channels != None:
            colors = [
                0xFF0000, # Red
                0x00FF00, # green
                0xFF00FF,  # Purple
                0xFFFF00   # Yellow
            ]
            self.channel_colors = colors[:len(self.channels)]
            
        # get_blosc_codec(writer_config['codec'], writer_config['clevel'])
    
    def read_multichannel_image(
        self,
        channel_paths:List[PathLike]
    ) -> dask.array.core.Array:
        """
            Reads image files and stores them in a dask array.
            
            channel_paths:List[PathLike]
                Paths where the images are located in multiple channels
            
            Returns
            ------------------------
            dask.array.core.Array:
                Dask array with the images. Returns None if it was not possible to read the images.
        """
        
        image_channel = []
    
        if len(channel_paths) <= 1:
            return None
        
        for channel_path in channel_paths:
            print("- Reading channel: ", Path(channel_path).stem)
            image_channel.append(
                self.read_channel_image(channel_path)
            )
        
        # try:
        #     image_channel = stack(image_channel, axis=0)
        # except ValueError as err:
        #     return None
        
        return image_channel
    
    def read_channel_image(
        self,
        path:PathLike
    ) -> dask.array.core.Array:
    
        """
            Reads image files and stores them in a dask array.
            
            path:PathLike
                Path where the images are located
            
            Returns
            ------------------------
            dask.array.core.Array:
                Dask array with the images. Returns None if it was not possible to read the images.
        """
        
        images = None
        
        try:
            # Path comes with '/' then just adding the file name and format
            filename_pattern = f'{path}*.{self.file_format}*'
            images = imread(filename_pattern)
            
        except ValueError:
            raise ValueError('- No images found in ', path)
        
        return images
    
    def pad_array_n_d(self, arr:ArrayLike, dim:int=5) -> ArrayLike:
    
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

    def compute_pyramid(
        self,
        data:dask.array.core.Array, 
        n_lvls:int, 
        scale_axis:Tuple[int]
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
            preserve_dtype=True
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
                "kwargs": {} # No extra parameters were used different from the orig. array and scales
            }
        }
    
    def convert(
        self,
        writer_config:dict,
        image_name:str='zarr_multiscale.zarr',
        # chunks:List[int]=[1, 736, 560]
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
        dask.config.set(
            {
                'temporary-directory': self.dask_folder,
                'local_directory': self.dask_folder,
                'tcp-timeout': '60s',
                'array.chunk-size': '384MiB',
                'distributed.comm.timeouts': {
                    'connect': '60s', 
                    'tcp': '60s'
                },
                'distributed.scheduler.bandwidth': 100000000,
		        'distributed.worker.memory.rebalance.measure': 'managed_in_memory',
                'distributed.worker.memory.target': False,
                'distributed.worker.memory.spill': False,
                'distributed.worker.memory.pause': False,
                'distributed.worker.memory.terminate': False
                # 'distributed.scheduler.unknown-task-duration': '15m',
                # 'distributed.scheduler.default-task-durations': '2h',
            }
        )
        
        # print(dask.config.config)
        
        cluster = LocalCluster()
        # cluster.adapt(
        #     minimum=1, maximum=4, interval='10s', target_duration='60s'
        # )
        client = Client(cluster)
        
        # print(client.scheduler_info())
        regex_files = f'[^"]*.{self.file_format}*'
        regex_files = "({})".format(regex_files)

        # Looking for channels
        level_folder = search_images_in_folder(self.input_data, regex_files)
        glob_search = '/*'*level_folder
        
        # Listing channels with '/' # TODO Works with only one dir
        list_channels = glob(str(self.input_data) + glob_search + '/')

        print("Check reads: ", list_channels)
        image = self.read_multichannel_image(list_channels)
        
        # Reading single channel
        if not isinstance(image, dask.array.core.Array) and not isinstance(image, list):
            image = self.pad_array_n_d(
                self.read_channel_image(list_channels[0]),
                4
            )
        
        scale_axis = []
        for axis in range(len(image[0].shape)-len(writer_config['scale_factor'])):
            scale_axis.append(1)
        
        scale_axis.extend(list(writer_config['scale_factor']))
        scale_axis = tuple(scale_axis)
        
        with performance_report(filename="dask-report.html"):
            for idx in range(len(image)):
                pyramid_data = self.compute_pyramid(
                    image[idx], 
                    writer_config['pyramid_levels'],
                    scale_axis
                )
                
                pyramid_data = [self.pad_array_n_d(pyramid) for pyramid in pyramid_data]
                image_name = self.channels[idx] + '.zarr' if self.channels else image_name
                channel_names = [self.channels[idx]] if self.channels else None
                channel_colors = [self.channel_colors[idx]] if self.channel_colors else None
                
                dask_jobs = self.writer.write_multiscale(
                    pyramid=pyramid_data,  # : types.ArrayLike,  # must be 5D TCZYX
                    image_name=image_name,  #: str,
                    physical_pixel_sizes=self.physical_pixels,
                    channel_names=channel_names,#['CH_0', 'CH_1', 'CH_2', 'CH_3'],
                    channel_colors=channel_colors,
                    scale_factor=scale_axis,  # : float = 2.0,
                    chunks=pyramid_data[0].chunksize,#chunks,#writer_config['chunks'],
                    storage_options=self.opts,
                    compute_dask=True,
                    **self.get_pyramid_metadata()
                )

                if len(dask_jobs):
                    dask_jobs = dask.persist(*dask_jobs)#, get=dask.threaded.get)
                    # dask_jobs = dask_jobs.persist()
                    progress(dask_jobs)
        
        client.close()

def main():
    default_config = get_default_config()

    mod = ArgSchemaParser(
        input_data=default_config,
        schema_type=ZarrConvertParams
    )
    
    args = mod.args
    
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)
    
    zarr_converter = ZarrConverter(
        input_data=args['input_data'],
        output_data=args['output_data'], 
        blosc_config={
            'codec': args['writer']['codec'], 
            'clevel': args['writer']['clevel']
        },
        channels=['CH_1', 'CH_2'],
        physical_pixels=[2.0, 1.8, 1.8]
    )
    
    start_time = time.time()
    
    zarr_converter.convert(args['writer'])
    
    end_time = time.time()
    
    LOGGER.info(
        f"Done converting dataset. Took {end_time - start_time}s."
    )

if __name__ == "__main__":
    main()