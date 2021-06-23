mkdir data
curl --output data/train.jsonl https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/text-to-code/dataset/concode/train.json
curl --output data/dev.jsonl https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/text-to-code/dataset/concode/dev.json
curl --output data/test.jsonl https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Text-Code/text-to-code/dataset/concode/test.json