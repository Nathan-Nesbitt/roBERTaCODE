# load in the data
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


class Data:
    def __init__(self, languages):

        self.languages = languages

        # Creates the data directory
        try:
            os.mkdir(os.path.dirname("./data/"))
        except FileExistsError:
            pass

        # Handles all conditions for the parameters
        if self.languages == [None]:
            self.languages = ["python", "php", "java", "go", "javascript", "ruby"]

        elif not set(self.languages).issubset(
            set(["python", "php", "java", "go", "javascript", "ruby"])
        ):
            sys.stderr.write(
                f"Please specify a real language in languages: {languages}\n"
            )
        with Pool(6) as p:
            input = []
            for i in range(len(self.languages)):
                input.append((self.languages[i], i))
            p.starmap(download_file, input)

    def download_file(self, language, position):
        location = (
            f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"
        )
        # Get the file from the server
        requested_file = urlopen(location)
        total_length = requested_file.headers.get("content-length")
        if total_length:
            total_length = int(total_length)
            blocksize = max(4096, total_length // 100)
        else:
            blocksize = 1000000
        with open(f"/tmp/{language}.zip", "wb") as tempzip:
            with tqdm(total=total_length, position=position) as tq:
                tq.set_description(f"Downloading {language}")
                while True:
                    data = requested_file.read(blocksize)
                    if not data:
                        break
                    tempzip.write(data)
                    tq.update(blocksize)
        with ZipFile(f"/tmp/{language}.zip") as zf:
            zf.extractall(path="./data/")
        # Finally delete the temp file
        os.remove(f"/tmp/{language}.zip")
        # Get all of the zipped files and extract them
        files = []
        for file in glob.glob(f"data/{language}/**/*.gz", recursive=True):
            files.append(file)
        with tqdm(total=len(files), position=position) as tq:
            tq.set_description(f"Unzipping {language}")
            for i in files:
                gunzip(i)
                tq.update(1)
            tq.close()


if __name__ == "__main__":
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
    languages = a
    Data(languages)
