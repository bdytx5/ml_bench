
#!/usr/bin/env python

import argparse
import multiprocessing
import os
import subprocess
from typing import List, Tuple

import _load_profiles
import _timing
import numpy
import mlflow

VERSION: str = "v1-2024-04-11-0"
BENCH_OUTFILE: str = "bench_mlflow.csv"
BENCH_FIELDS: Tuple[str] = (
    "test_name",
    "test_profile",
    "test_variant",
    "client_version",
    "client_type",
    "server_version",
    "server_type",
)
TIMING_DATA: List = []

def get_mlflow_version():
    try:
        result = subprocess.run(
            ["pip", "show", "mlflow"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith("Version:"):
                return line.split(":")[1].strip()
        return "Version not found"
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to get mlflow version: {e}")
        return "Error"

def run_one(args, n=0, m=0):
    print(f"Running with mode: {args.mode}")  # Debugging statement

    if args.mode == "online":
        mlflow.set_tracking_uri("http://mlflow_server:5000")  # Set this to your MLflow server URL
    else:
        mlflow.set_tracking_uri("file:///tmp/mlruns")  # Local file-based storage

    mlflow.start_run(run_name=args.test_name)

    for e in range(args.num_history):
        metrics = {}
        params = {}
        artifacts = []

        for i in range(args.history_floats):
            metrics[f"f_{i}"] = float(n + m + e + i)
        for i in range(args.history_ints):
            metrics[f"n_{i}"] = n + m + e + i
        for i in range(args.history_strings):
            params[f"s_{i}"] = str(n + m + e + i)
        for i in range(args.history_tables):
            table_data = [[n + m, e, i, i + 1]]
            artifact_path = f"table_{i}.csv"
            numpy.savetxt(artifact_path, table_data, delimiter=",", fmt='%d')
            artifacts.append(artifact_path)
        for i in range(args.history_images):
            image_data = numpy.random.randint(
                255,
                size=(args.history_images_dim, args.history_images_dim, 3),
                dtype=numpy.uint8,
            )
            artifact_path = f"image_{i}.png"
            from PIL import Image
            Image.fromarray(image_data).save(artifact_path)
            artifacts.append(artifact_path)

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        for artifact_path in artifacts:
            mlflow.log_artifact(artifact_path)

    mlflow.end_run()

def run_sequential(args, m=0):
    for n in range(args.num_sequential):
        run_one(args, n, m)

def run_parallel(args):
    procs = []
    for n in range(args.num_parallel):
        p = multiprocessing.Process(
            target=run_sequential, args=(args, n * args.num_parallel)
        )
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()

def setup(args):
    pass

def teardown(args):
    pass

@_timing.timeit(TIMING_DATA)
def time_load(args):
    if args.num_parallel > 1:
        run_parallel(args)
    else:
        run_sequential(args)

def run_load(args):
    setup(args)
    time_load(args)
    teardown(args)

def main():
    parser = argparse.ArgumentParser(description="benchmark mlflow performance")
    parser.add_argument("--test_name", type=str, default="")
    parser.add_argument(
        "--mode", type=str, choices=("online", "offline"), default="offline"
    )
    parser.add_argument(
        "--test_profile", type=str, default="", choices=list(_load_profiles.PROFILES)
    )
    parser.add_argument("--test_variant", type=str, default="")
    parser.add_argument("--server_version", type=str, default="")
    parser.add_argument("--server_type", type=str, default="")
    parser.add_argument("--client_version", type=str, default=get_mlflow_version())
    parser.add_argument("--client_type", type=str, default="")
    parser.add_argument("--num_sequential", type=int, default=1)
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--num_history", type=int, default=1)
    parser.add_argument("--history_floats", type=int, default=0)
    parser.add_argument("--history_ints", type=int, default=0)
    parser.add_argument("--history_strings", type=int, default=0)
    parser.add_argument("--history_tables", type=int, default=0)
    parser.add_argument("--history_images", type=int, default=0)
    parser.add_argument("--history_images_dim", type=int, default=16)
    parser.add_argument("--project_name", type=str, default="default_project")
    parser.add_argument("--workspace", type=str, default="default_workspace")
    parser.add_argument("--use-spawn", action="store_true")

    args = parser.parse_args()
    # required by golang experimental client when testing multiprocessing workloads
    if args.use_spawn:
        multiprocessing.set_start_method("spawn")

    args_list = []
    if args.test_profile:
        args_list = _load_profiles.parse_profile(parser, args, copy_fields=BENCH_FIELDS)
    else:
        args_list.append(args)

    for args in args_list:
        print(f"Parsed arguments: {args}")  # Debugging statement
        run_load(args)
        prefix_list = [VERSION]
        for field in BENCH_FIELDS:
            prefix_list.append(getattr(args, field))
        _timing.write(BENCH_OUTFILE, TIMING_DATA, prefix_list=prefix_list)

if __name__ == "__main__":
    main()