"""General argument parsing for all scripts"""

import argparse


def cli_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="configuration file")
    parser.add_argument(
        "-r",
        "--results",
        type=str,
        default=None,
        help="override config file results dir",
    )
    args = parser.parse_args(args)

    return args
