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
