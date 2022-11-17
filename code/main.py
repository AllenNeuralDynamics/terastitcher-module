import sys

from terastitcher_module import terastitcher
from utils import utils


def main() -> None:
    output_folder = terastitcher.main()
    bucket_path = sys.argv[-1]

    print(f"Bucket path: {bucket_path} - Output path: {output_folder}")
    # Copying output to bucket
    dataset_name = output_folder.replace("/scratch/", "")
    s3_path = f"s3://{bucket_path}/{dataset_name}"
    for out in utils.execute_command_helper(
        f"aws s3 mv --recursive {output_folder} {s3_path}"
    ):
        print(out)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {s3_path}",
        "/root/capsule/results/output_stitching.txt",
    )


if __name__ == "__main__":
    main()
