"""
    A script for downloading and cleanly installing the CodeSearchNet
    database for model training. Allows for partial or complete installation
    based on provided parameters.

    Nathan Nesbitt 2020
"""

import argparse
import requests
import zipfile
from zipfile import ZipFile
from io import BytesIO
import os
import sys
import gzip
from pathlib import Path
import json

parser = argparse.ArgumentParser(
    prog="load_data.py", description="Downloads dataset from CodeSearchNet"
)

parser.add_argument(
    "language",
    metavar="lang",
    type=str,
    nargs="?",
    help="Indicates Language Type for Download. \
        Can be python, php, java, go, javascript or ruby",
)


def get_data(language):
    # The file locations as defined in the readme
    filename = (
        "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/" + language + ".zip"
    )

    try:
        os.mkdir("./data/" + language)
    except FileExistsError:
        pass

    # Get the file from the server
    requested_file = requests.get(filename, stream=True)
    # Unzip it
    zipped_files = ZipFile(BytesIO(requested_file.content), "r")

    for zipped_file in zipped_files.filelist:
        # If the file is a folder
        if zipped_file.is_dir():
            try:
                os.mkdir("./data/" + language)
            except FileExistsError:
                pass
        # Else we are dealing with a file
        else:
            # Creates path
            Path("./data/" + zipped_file.filename).parent.absolute().mkdir(
                parents=True, exist_ok=True
            )
            # Deals with gzipped file extentions
            file_path = "./data/" + zipped_file.filename
            if file_path[-2:] == "gz":
                file_path = file_path[:-3]
            # Opens and writes the file
            with open(file_path, "wb+") as write_file:
                f = zipped_files.read(zipped_file.filename)
                # Because we have double zipped files in here, we decompress
                if zipped_file.filename[-2:] == "gz":
                    write_file.write(gzip.decompress(f))
                else:
                    write_file.write(f)


# Parses the arguments passed to the script
args = parser.parse_args()

# Creates the data directory
try:
    os.mkdir(os.path.dirname("./data/"))
except FileExistsError:
    pass

languages = ["python", "php", "java", "go", "javascript", "ruby"]

# Handles all conditions for the parameters
if not args.language:
    for i in languages:
        get_data(i)
elif args.language not in languages:
    sys.stderr.write(
        "Please specify a real language in languages: " + str(languages) + "\n"
    )
else:
    get_data(args.language)
