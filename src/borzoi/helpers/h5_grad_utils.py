import h5py
import numpy as np
import argparse
import subprocess
import tempfile
import os
from baskerville.helpers.gcs_utils import download_from_gcs, upload_file_gcs


def collect_h5_borzoi(out_dir, num_procs, sad_stat) -> None:
    h5_file = "scores_f0c0.h5"

    # count sequences
    num_seqs = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5_file)
        job_h5_open = h5py.File(job_h5_file, "r")
        num_seqs += job_h5_open[sad_stat].shape[0]
        seq_len = job_h5_open[sad_stat].shape[1]
        num_targets = job_h5_open[sad_stat].shape[-1]
        job_h5_open.close()

    # initialize final h5
    final_h5_file = "%s/%s" % (out_dir, h5_file)
    final_h5_open = h5py.File(final_h5_file, "w")

    # keep dict for string values
    final_strings = {}

    job0_h5_file = "%s/job0/%s" % (out_dir, h5_file)
    job0_h5_open = h5py.File(job0_h5_file, "r")
    for key in job0_h5_open.keys():
        key_shape = list(job0_h5_open[key].shape)
        key_shape[0] = num_seqs
        key_shape = tuple(key_shape)
        if job0_h5_open[key].dtype.char == "S":
            final_strings[key] = []
        else:
            final_h5_open.create_dataset(
                key, shape=key_shape, dtype=job0_h5_open[key].dtype
            )

    # set values
    si = 0
    for pi in range(num_procs):
        # open job
        job_h5_file = "%s/job%d/%s" % (out_dir, pi, h5_file)
        job_h5_open = h5py.File(job_h5_file, "r")

        # append to final
        for key in job_h5_open.keys():
            job_seqs = job_h5_open[key].shape[0]
            if job_h5_open[key].dtype.char == "S":
                final_strings[key] += list(job_h5_open[key])
            else:
                final_h5_open[key][si : si + job_seqs] = job_h5_open[key]

        job_h5_open.close()
        si += job_seqs

    # create final string datasets
    for key in final_strings:
        final_h5_open.create_dataset(key, data=np.array(final_strings[key], dtype="S"))

    final_h5_open.close()


def download_h5_gcs(output_gcs_dir, num_processes) -> str:
    temp_dir = tempfile.mkdtemp()  # create a temp dir for output
    print(f"temp_dir is {temp_dir}")
    out_dir = temp_dir + "/" + output_gcs_dir.split("/")[-1]
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    # download output from tempfile
    for pi in range(num_processes):
        if not os.path.isdir(f"{out_dir}/job{pi}"):
            os.mkdir(f"{out_dir}/job{pi}")
        download_from_gcs(
            f"{output_gcs_dir}/job{pi}/scores_f0c0.h5",
            f"{out_dir}/job{pi}/scores_f0c0.h5",
        )
        print(f"Done downloading {pi} partition")
    # download all of the files in the folder
    # Use gsutil to copy the contents recursively
    # subprocess.check_call(["gsutil", "-m", "cp", "-r", output_gcs_dir, temp_dir])
    print(f"outdir is {out_dir}")
    print(f"gcs_out_dir is {output_gcs_dir}")
    print(f"Done dowloading")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Process and collect h5 files.")

    parser.add_argument(
        "out_dir", type=str, help="Output directory for processed data."
    )
    parser.add_argument("num_procs", type=int, help="Number of processes to use.")
    parser.add_argument("sad_stat", type=str, help="Stats to concatenate. E.g. grads")
    parser.add_argument(
        "--gcs",
        action="store_true",
        help="Flag indicating if the file is on Google Cloud Storage.",
    )

    args = parser.parse_args()
    if args.gcs:
        # download files to tempdir
        local_out_dir = download_h5_gcs(args.out_dir, args.num_procs)
    collect_h5_borzoi(local_out_dir, args.num_procs, args.sad_stat)
    # upload to gcs
    print(f"is there such a file? {local_out_dir}/scores_f0c0.h5")
    print(os.path.isfile(f"{local_out_dir}/scores_f0c0.h5"))
    upload_file_gcs(f"{local_out_dir}/scores_f0c0.h5", args.out_dir)


if __name__ == "__main__":
    main()
