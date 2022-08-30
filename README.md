# TeraStitcher Module

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

Striping and stitching module for 3D teravoxel-sized microscopy images using  [TeraStitcher](https://github.com/abria/TeraStitcher). This module currently supports Windows and Linux based OS as well as VM in GCloud using GSCFUSE.

## Installation on GCP VertexAI
You need to create a User-Managed Notebook with `Python 3`. If you want to add GPU processing, please select `Python 3 (CUDA Toolkit 11.0)` or with a superior available Toolkit. We recommend a GPU P100 device or Titan V(olta) if available.

Prior to the installation, download the lastest version of TeraStitcher with support for BioFormats from its website. As today, the most recent version this is the most recent version of [TeraStitcher with BF](https://unicampus365-my.sharepoint.com/:u:/g/personal/g_iannello_unicampus_it/EeOijoWwL9xBheoGjty4KU4BlsV_x1iJk7hJTc38OFwWsg?e=Kjbfr4).

Create a new conda environment and execute the following commands:

```
$ conda create -n stitching -y python=3.8
$ conda activate stitching
$ conda install -y ipykernel
```

Then, open a terminal and clone this repository in the virtual machine and place the dowloaded `.tar.gz` file in the following path: `terastitcher-module/environment/GCloud/`. Now, execute the installer (this will take a couple of minutes):

```
$ cd terastitcher-module/environment/GCloud/
$ chmod +x terastitcher_installer.sh
$ ./terastitcher_installer.sh
```

You should be able to run terastitcher and pystripe with MPI. To verify the installation close the current terminal and execute the following commands in a new one:

```
$ terastitcher -h
```

```
$ pystripe -h
```

```
$ mpirun -V
```

## Local installation
Follow the steps provided [here](https://github.com/abria/TeraStitcher/wiki/Download-and-install) to install TeraStitcher for your platform. Please, install the command-line version with BioFormats support [Link](https://github.com/abria/TeraStitcher/wiki/Binary-packages#terastitcher-portable-with-support-for-bioformats-command-line-version).

After TeraStitcher is correctly installed in your platform, create a new conda environment and execute the following commands:

```
$ conda create -n stitch -y python=3.8
$ conda activate stitch
```

Then, clone the repository and install the packages using pyproject.toml file.

```
$ git clone https://github.com/AllenNeuralDynamics/terastitcher-module
$ cd terastitcher-module
$ pip install -e .
```

Afterwards, set --parastitcher_path parameter to the Parastitcher python script. You can do this in the command line or directly in the code. It should be located in `YOUR_DIRECTORY/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/Parastitcher.py`.

## Parameters
This module uses the following principal parameters:
- --input_data: Path where the data is located. If it's located in a GCS bucket, please refer to it such as: `gs://bucket-name/dataset_name/image_folder`.
- --output_data: Output path. If it's located in a GCS bucket, please refer to it such as: `gs://bucket-name/output_path`.

By default, this modules loads the parameters used in the stitching process based on the GUI software developed in lab 440. To check the default values, please execute:
```
$ python terastitcher.py --help
```

This will be helpful if you want to overwrite them. 

**Note**: It is important to mention that this module uses different tools that output processed images. For erasing streaks in EM images, we are using pystripe which outputs the same dataset folder structure with the processed images. Terastitcher does the same but with a different folder structure and finally we convert the stitched images to OME-Zarr. Therefore, **you need 3X disk space based on your original disk size**. By default, we have a flag `clean_output` which is set to True. When this flag is set to True, it deletes the intermediate images generated by this module leaving only the OME-Zarr output.

If you run this pipeline in GCP, this module will automatically load the bucket(s) using GCSFUSE. As today, there are some performance limitations when using GCSFUSE with MPI if your dataset contains many small images. The performance will improve if you have larger files containing these small images (e.g. 3D stacks of images).

### Configuration file
Since TeraStitcher has multiple steps, we created a configuration dictionary with the parameters for each step. You can add more parameters based on terastitcher documentation. You can check the default parameters in `terastitcher-module/code/src/params.py`. For example:

```
"import_data" : {
    // Parameters that have values
    "ref1":"H",
    "ref2":"V",
    "ref3":"D",
    "vxl1":"1.800",
    "vxl2":"1.800",
    "vxl3":"2",
    // Flag parameters
    "additional_params": [
        "sparse_data",
        "libtiff_uncompress",
        'rescan'
    ]
}
```

The align and merge steps are computationally expensive. Therefore, we added cpu parameters for executing these steps. For example:

```
"align" : {
    "cpu_params": {
        // Set estimate processes True if you are not sure what's the best number of workers for your CPU or the image depth you have
        "estimate_processes": false,
        "image_depth": 4200,
        "number_processes": 16, //16 cores
        // Hostfile for mpirun, it is not considered on Windows OS
        "hostfile": "/home/jupyter/terastitcher-module/environment/GCloud/hostfile",
        // Additional mpi params, these work only on Linux based distributions 
        "additional_params": [
            "use-hwthread-cpus",
            "allow-run-as-root"
        ]
    },
    // terastitcher's align parameters
    "subvoldim": 100
}
```

### Execution example
```
$ cd ~
$ python terastitcher-module/code/src/terastitcher.py --input_data gs://bucket-name/dataset/images --output_data gs://bucket-name/dataset_stitched
```

## TeraStitcher Documentation
You can download TeraStitcher documentation from [here](https://unicampus365-my.sharepoint.com/:b:/g/personal/g_iannello_unicampus_it/EYT9KbapjBdGvTAD2_MdbKgB5gY_h9rlvHzqp6mUNqVhIw?e=s8GrFC)

## Pystripe Documentation
You can access Pystripe documentation from [here](https://github.com/chunglabmit/pystripe)