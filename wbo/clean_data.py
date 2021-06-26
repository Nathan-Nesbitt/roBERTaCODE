import json
from tqdm import tqdm


def clean_data(location, length, position):
    """
    Iterates through the input data file adding the source code functions into
    dataValues and the labels into dataLabels

    Makes Sure to seperate [".", "[", "]", "(", ")", "{", "}" ":"]
    Makes sure to keep ["+", "-", "*", "/", "=", "%", "!", "<", ">"]
      together if they are beside eachother (ie. += )
    Otherwise seperates ["+", "-", "*", "/", "=", "%", "!", "<", ">"]

    This is an optimized version of the previous function.
    """
    keep = ["+", "-", "*", "/", "=", "%", "!", "<", ">"]
    separate = [".", "[", "]", "(", ")", "{", "}" ":"]

    reading_file = open(location, "r", encoding="utf-8")
    values_file = open(f"{location}.values", "a")

    with tqdm(total=length, position=position) as tq:
        tq.set_description(f"Parsing {location}")
        while True:
            line = reading_file.readline()
            if line == "":
                break
            jsonObject = json.loads(line)
            seperatedFunction = ""
            for element in range(0, len(jsonObject["function"])):
                if jsonObject["function"][element] in separate:
                    seperatedFunction += f" {jsonObject['function'][element]} "
                elif jsonObject["function"][element] in keep:
                    if (element > 0) and jsonObject["function"][
                        element - 1
                    ] not in keep:
                        seperatedFunction += " "

                    seperatedFunction += jsonObject["function"][element]

                    if (element + 1) < len(jsonObject["function"]) and jsonObject[
                        "function"
                    ][element + 1] not in keep:
                        seperatedFunction += " "
                else:
                    seperatedFunction += jsonObject["function"][element]
            if jsonObject["label"] == "Correct":
                values_file.write(
                    json.dumps({"input_ids": seperatedFunction, "labels": 1}) + "\n"
                )
            else:
                values_file.write(
                    json.dumps({"input_ids": seperatedFunction, "labels": 0}) + "\n"
                )
            tq.update(1)
    reading_file.close()
    values_file.close()


"""Merges the Wrong Binary Operator Data from the ETH Py150 Open Dataset"""

directory = "./data"

# Open output directory
position = 0
for dir in ["train", "test", "validation"]:
    size = 0
    with open(f"{directory}/{dir}/full.txt", "a+", encoding="utf-8") as text_file:

        # For each input file
        for i in range(0, 4):
            file = f"{dir}{i}.json"
            path = f"{directory}/{dir}/{file}"
            print(f"Parsing: {path}")

            # Open input file
            with open(path, "r") as json_file:
                for line in json_file:
                    # Writes each json object to output file
                    text_file.write(line)
                    size += 1
    location = f"./data/{dir}/full.txt"
    clean_data(location=location, length=size, position=position)
    position += 1
