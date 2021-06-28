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
from pathlib import Path
import math
import json
from tokenizers import ByteLevelBPETokenizer


class Data:
    """
    Class that provides the data functionality for working with the
    CodeSearchNet dataset.
    """

    def __init__(self, languages=[None], sizes=[None], no_text=False):
        """
        Languages is the set of languages to download. If left blank
        it defaults to all languages. The options for languages is
        "python", "php", "java", "go", "javascript", "ruby".

        Sizes is the possible sizes of the datasets to generate, the options
        are small, medium, and large. Leaving this blank defaults to all
        three.

        no_text sets it so that we only generate data files with code, no
        text.
        """

        self.languages = languages
        self.sizes = sizes
        self.no_text = no_text

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
            raise NameError(
                f'This language is not supported. Please choose from one of the following: "python", "php", "java", "go", "javascript", "ruby"'
            )

        if self.sizes == [None]:
            self.sizes = ["small", "medium", "large"]
        elif not set(self.sizes).issubset(set(["small", "medium", "large"])):
            raise NameError(
                f'This size is not supported. Please choose from one of the following: "small", "medium", "large"'
            )

    def download_data(self):
        """
        Script that fetches the CodeSearchNet dataset and unzips all of the
        children.
        """
        with Pool(6) as p:
            input = []
            for i in range(len(self.languages)):
                input.append((self.languages[i], i))
            p.starmap(self.download_file, input)

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

    def clean_data(self, lang=None, train=True):
        """
        Cleans the datasets so they fit the line by line input for the
        script. This takes 2 possible args

        lang allows you to select individual language that you want to
        generate datasets for. By default this is set to all of the files
        so you can generate a full model based on all langauges. But you
        can also specify a single language.

        train allows you to generate a dev dataset instead of a training
        dataset. If you are trying to do early stopping on a fully trained
        model you will need this.
        """

        combined = False
        if not lang:
            lang = self.languages
            combined = True
        else:
            if lang not in self.languages:
                raise NameError(
                    f"This language is not supported. Please choose from one of the following: {self.languages}"
                )
            lang = [lang]

        input_files = {}
        languages = {}

        # Read in the files that can be worked with based on the language
        for language in lang:
            languages[language] = {}
            files = []
            if train:
                location = f"data/{language}/final/jsonl/train"
            else:
                location = f"data/{language}/final/jsonl/valid"
            for path in Path(location).glob("*.jsonl"):
                files.append(f"{location}/{path.name}")
            input_files[language] = files

        # Calculate the required sizes for small (SQRT(n)), medium (n/2), and large (full)
        for language in lang:
            for size in self.sizes:
                if size == "small":
                    languages[language][size] = round(
                        math.sqrt(len(input_files[language]))
                    )
                if size == "medium":
                    languages[language][size] = round(len(input_files[language]) / 2)
                if size == "large":
                    languages[language][size] = round(len(input_files[language]))

        # Read in the files
        for language, files in input_files.items():
            for size, n_files in languages[language].items():
                if combined:
                    if train:
                        output_file = open(f"data/train_combined_{size}.txt", "a")
                    else:
                        output_file = open(f"data/valid_combined_{size}.txt", "a")
                else:
                    if train:
                        output_file = open(f"data/train_{language}_{size}.txt", "a")
                    else:
                        output_file = open(f"data/valid_{language}_{size}.txt", "a")

                with tqdm(total=n_files) as tq:
                    tq.set_description(f"Generating {size} {language} dataset")
                    for i in range(n_files):
                        with open(files[i], "r") as input_file:
                            for line in input_file:
                                json_data = json.loads(line)
                                # We set the input to be (512 - 3) / 2 tokens to satisfy BERT architechture
                                n = len(json_data["docstring_tokens"])
                                if n > 254:
                                    n = 254
                                if self.no_text:
                                    n = 0
                                output_file.write(
                                    " ".join(
                                        [
                                            "".join(k)
                                            for k in json_data["docstring_tokens"][:n]
                                        ]
                                    )
                                )
                                output_file.write(" ")
                                output_file.write(
                                    " ".join(
                                        [
                                            "".join(k)
                                            for k in json_data["code_tokens"][: 509 - n]
                                        ]
                                    )
                                )
                                output_file.write("\n")
                            tq.update(1)
                tq.close()

    def generate_tokenizer(self, lang=None):

        if not lang:
            lang = "combined"
        else:
            if lang not in self.languages:
                raise NameError(
                    f"This language is not supported. Please choose from one of the following: {self.languages}"
                )

        for size in self.sizes:
            tokenizer = ByteLevelBPETokenizer(lowercase=False)

            tokenizer.train(
                files=f"data/train_{lang}_{size}.txt",
                vocab_size=32000,
                min_frequency=3,
                special_tokens=[
                    "<s>",
                    "<pad>",
                    "</s>",
                    "<unk>",
                    "<mask>",
                ],
            )

            os.makedirs(f"tokenizer_{lang}_{size}", exist_ok=True)
            tokenizer.save_model(f"tokenizer_{lang}_{size}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="data.py",
        description="Transforms the data into an acceptable format for pre-training",
    )

    parser.add_argument(
        "--languages",
        "-l",
        metavar="languages",
        type=str,
        nargs="+",
        help="This is the language you would like to pre-train on. 'all' creates a full dataset.",
        choices=["python", "java", "javascript", "go", "ruby", "php", "all"],
    )

    parser.add_argument(
        "--sizes",
        "-s",
        metavar="sizes",
        type=str,
        nargs="+",
        help="The sizes of tokenizers and datasets that you would like to create. Leave empty for all.",
        choices=["small", "medium", "large"],
    )

    parser.add_argument(
        "--no_text",
        "-n",
        action="store_true",
        help="This only uses the programming code instead of the programming code and the text. \
            As this was not done in the original paper it is added as an extra parameter, but it \
            can be used to test if code alone is enough to improve performance",
    )

    args = parser.parse_args()

    languages = args.languages

    if "all" in args.languages:
        languages = [None]
    if args.sizes == None:
        args.sizes = [None]

    # Parses the arguments passed to the script
    data = Data(languages=languages, sizes=args.sizes, no_text=args.no_text)

    if "all" in args.languages:
        # Downloads and unzipps the datasets
        data.download_data()
        # Cleans the train data
        data.clean_data()
        # Cleans the validation data
        data.clean_data(train=False)
        # Generates the tokenizer
        data.generate_tokenizer()
        args.languages.remove("all")

    for lang in args.languages:
        # Cleans the train data
        data.clean_data(lang=lang)
        # Cleans the validation data
        data.clean_data(lang=lang, train=False)
        # Generates the tokenizer
        data.generate_tokenizer(lang=lang)
