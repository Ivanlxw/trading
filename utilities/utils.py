import json
import os
import argparse

def remove_bs(s: str):
    # remove backslash at the end from reading from a stock_list.txt
    return s.replace("\n", "")


def load_credentials(credentials_fp):
    with open(credentials_fp, 'r') as f:
        credentials = json.load(f)
        for k, v in credentials.items():
            os.environ[k] = v


def parse_args():
    parser = argparse.ArgumentParser(description='Configs for running main.')
    parser.add_argument('-c', '--credentials', required=True,
                        type=str, help="credentials filepath")
    parser.add_argument('-n', '--name', required=False, default="",
                        type=str, help="name of backtest/live strat run")
    parser.add_argument('-b', '--backtest', required=False, type=bool, default=True,
                        help='backtest filters?')
    parser.add_argument('-f', '--fundamental', required=False,
                        type=bool, default=False, help="Use fundamental data or not")
    parser.add_argument('--data_dir', default="./data/daily",
                        required=False, type=str, help="filepath to dir of csv files")
    return parser.parse_args()