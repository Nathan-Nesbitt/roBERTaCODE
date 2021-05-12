"""
    A script for downloading and cleanly installing the CodeSearchNet
    database for model training. Allows for partial or complete installation
    based on provided parameters.

    Nathan Nesbitt 2020
"""

import argparse
from zipfile import ZipFile
import os
import sys
from urllib.request import urlopen
from multiprocessing import Process
from sh import gunzip
import glob
from tqdm import tqdm
from multiprocessing import Pool

def main():
    
    parser = argparse.ArgumentParser(
        prog="load_data.py", description="Downloads dataset from CodeSearchNet"
    )
    
    parser.add_argument(
        "--languages",
        "-l",
        metavar="lang",
        type=str,
        nargs="?",
        help="Indicates Language Type for Download. \
            Can be python, php, java, go, javascript or ruby. None downloads all.",
    )
    # Parses the arguments passed to the script
    args = parser.parse_args()

    a = []
    a.append(args.languages)
    args.languages = a

    # Creates the data directory
    try:
        os.mkdir(os.path.dirname("./data/"))
    except FileExistsError:
        pass

    languages = ["python", "php", "java", "go", "javascript", "ruby"]

    # Handles all conditions for the parameters
    if args.languages == [None]:
        args.languages = languages
        
    elif not set(languages).issubset(set(args.languages)):
        sys.stderr.write(
            "Please specify a real language in languages: {}\n".format(languages)
        )
    with Pool(6) as p:
        input = []
        for i in range(len(args.languages)):
            input.append((args.languages[i], i))
        p.starmap(download_file, input)

def download_file(language, position):
    location = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip".format(language)
    # Get the file from the server
    requested_file = urlopen(location)
    total_length = requested_file.headers.get('content-length')
    if total_length:
        total_length = int(total_length)
        blocksize = max(4096, total_length//100)
    else:
        blocksize = 1000000 
    with open("/tmp/{}.zip".format(language), "wb") as tempzip:
        with tqdm(total=total_length, position=position) as tq:
            tq.set_description("Downloading {}".format(language))
            while True:
                data = requested_file.read(blocksize)
                if not data:
                    break
                tempzip.write(data)
                tq.update(blocksize)
    with ZipFile("/tmp/{}.zip".format(language)) as zf:
        zf.extractall(path="./data/")
    # Finally delete the temp file 
    os.remove("/tmp/{}.zip".format(language))
    # Get all of the zipped files and extract them
    files = []
    for file in glob.glob("data/{}/**/*.gz".format(language), recursive=True):
        files.append(file)
    with tqdm(total=len(files), position=position) as tq:
        tq.set_description("Unzipping {}".format(language))
        for i in files:
            gunzip(i)
            tq.update(1)    
        tq.close()


if __name__ == "__main__":
    main()