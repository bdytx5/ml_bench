#!/usr/bin/env python

import argparse
import multiprocessing
import os
from typing import List, Tuple
import pandas as pd
import numpy as np
import neptune
from neptune.types import File
import _load_profiles
import _timing


VERSION: str = "v1-2024-04-11-0"
BENCH_OUTFILE: str = "bench_neptune.csv"
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

def run_one(args, n=0, m=0):
    print(f"Running with mode: {args.mode}")  # Debugging statement

    if args.mode == "online":
        run = neptune.init(project='your_workspace/your_project', api_token='YOUR_API_TOKEN')
    else:
        run = neptune.init(project='your_workspace/your_project', api_token='YOUR_API_TOKEN', mode='offline')

    for e in range(args.num_history):
        d = {}
        for i in range(args.history_floats):
            d[f"f_{i}"] = float(n + m + e + i)
        for i in range(args.history_ints):
            d[f"n_{i}"] = n + m + e + i
        for i in range(args.history_strings):
            d[f"s_{i}"] = str(n + m + e + i)
        for i in range(args.history_tables):
            table_data = pd.DataFrame([[n + m, e, i, i + 1]], columns=["a", "b", "c", "d"])
            run[f"t_{i}"].upload(File.as_html(table_data))
        for i in range(args.history_images):
            image_data = np.random.randint(
                255,
                size=(args.history_images_dim, args.history_images_dim, 3),
                dtype=np.uint8,
            )
            run[f"i_{i}"].append(File.as_image(image_data))

        for key, value in d.items():
            run[key].log(value)

    run.stop()

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
    parser = argparse.ArgumentParser(description="benchmark neptune performance")
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
    parser.add_argument("--client_version", type=str, default=neptune.__version__)
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
    parser.add_argument("--core", type=str, default="", choices=("true", "false"))
    parser.add_argument("--use-spawn", action="store_true")

    args = parser.parse_args()
    # required by golang experimental client when testing multiprocessing workloads
    if args.use_spawn:
        multiprocessing.set_start_method("spawn")

    args_list = []
    if args.test_profile:
        print("%"*100)
        print(args.mode)
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
