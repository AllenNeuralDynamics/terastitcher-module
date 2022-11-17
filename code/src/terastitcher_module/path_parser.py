import re
from pathlib import Path
from typing import List, Optional, Union

from utils import utils

PathLike = Union[str, Path]


class PathParser:
    @staticmethod
    def parse_path_gcs(
        input_path: PathLike, output_path: PathLike
    ) -> List[str]:
        """
        GCloud parser to know how many buckets we need to mount with gcsfuse.

        Parameters
        ------------------------
        input_path: PathLike
            Path where the data is located in the bucket.

        output_path: PathLike
            Path where the data will be saved in the same or a different bucket.

        Returns
        ------------------------
        List[str]:
            List with the bucket name(s).

        """
        gcs_regex = "gs://(.*?)/"

        gcs_bucket_input_re = re.match(gcs_regex, str(input_path))
        gcs_bucket_output_re = re.match(gcs_regex, str(output_path))

        if gcs_bucket_input_re == None or gcs_bucket_output_re == None:
            return []

        # Getting bucket names
        gcs_bucket_input = gcs_bucket_input_re.group(0)
        gcs_bucket_output = gcs_bucket_output_re.group(0)

        bucket_name_input = gcs_bucket_input.split("/")[2]
        bucket_name_output = gcs_bucket_output.split("/")[2]

        # Getting folder names for mount only dir
        folder_name_input = input_path[gcs_bucket_input_re.end() :]
        folder_name_output = output_path[gcs_bucket_output_re.end() :]

        bucket_input_config = {
            # "only-dir": folder_name_input,
            "max-conns-per-host": 20,
            "stat-cache-ttl": "5m0s",
            "max-retry-sleep": "5m0s",
            "additional_params": ["implicit-dirs", "disable-http2"],
        }

        bucket_output_config = {
            # "only-dir": folder_name_output,
            "max-conns-per-host": 20,
            "stat-cache-ttl": "5m0s",
            "max-retry-sleep": "5m0s",
            "additional_params": ["implicit-dirs", "disable-http2"],
        }

        if gcs_bucket_input == gcs_bucket_output:
            # Same parent, mount only one bucket

            utils.create_folder(bucket_name_input)
            utils.gscfuse_mount(
                bucket_name=bucket_name_input, params=bucket_input_config
            )

        else:
            # Mounting buckets in new folders
            utils.create_folder(bucket_name_input)
            utils.gscfuse_mount(
                bucket_name=bucket_name_input, params=bucket_input_config
            )

            utils.create_folder(bucket_name_output)
            utils.gscfuse_mount(
                bucket_name=bucket_name_output, params=bucket_output_config
            )

        return [bucket_name_input, bucket_name_output]
