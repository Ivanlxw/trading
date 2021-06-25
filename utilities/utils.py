import json
import os


def remove_bs(s: str):
    # remove backslash at the end from reading from a stock_list.txt
    return s.replace("\n", "")


def load_credentials(credentials_fp):
    with open(credentials_fp, 'r') as f:
        credentials = json.load(f)
        for k, v in credentials.items():
            os.environ[k] = v
