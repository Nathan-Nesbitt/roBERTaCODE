import json

"""Merges the Wrong Binary Operator Data from the ETH Py150 Open Dataset"""

directory = "./data"

# Open output directory
for dir in ["train", "test", "validation"]:
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


def clean_data(location):
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
                if (element > 0) and jsonObject["function"][element - 1] not in keep:
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
                json.dumps({"function": seperatedFunction, "labels": "1"}) + "\n"
            )
        else:
            values_file.write(
                json.dumps({"function": seperatedFunction, "labels": "0"}) + "\n"
            )
    reading_file.close()
    values_file.close()


train_location = "./data/train/full.txt"
validation_location = "./data/validation/full.txt"
clean_data(train_location)
clean_data(validation_location)
