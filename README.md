# RoBERTaCODE - Pre-Training RoBERTa on Code Summarization
This is the code used to train the model used in my (Nathan Nesbitt) Honours thesis
on code summarization using Transformer PTMs. It can be used to pre-train a 
roBERTa model using code, then fine-tuning it on the task of code summarization.

The fine-tuning in this repository is based on the CodeBERT repository. 

The original thesis code has been re-written to make it easier to understand and
run.

If you do not understand what this introduction means, read the [TLDR](#TLDR) 
which contains a summary of the functionality.

The fine-tuned models achieved the following scores:

| Model                  | Language   | Smoothed BLEU | BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 | METEOR | ROUGE_L | CIDEr |
| ------------------     |:----------:| :------------:| :-----:| :-----:| :-----:| :-----:| :-----:| :------:| -----:|
| RoBERTaCODE (Java)     |    Java    |     17.53     |  24.34 |  13.90 |  8.49  |  5.49  |  19.38 |  34.27  | 0.85  |
| RoBERTaCODE (Python)   |   Python   |     18.35     |  24.80 |  13.36 |  8.02  |  5.20  |  19.44 |  34.47  | 0.88  |
| RoBERTaCODE (All)      |    Java    |     17.65     |  24.86 |  14.46 |  9.06  |  6.01  |  19.40 |  33.95  | 0.91  |



## Requirements
- Python3
- A GPU
- 20GB or so of storage. (Depending on what dataset you download)

## Steps

The default scripts are configured to pre-train similar to the thesis. There
are 3 main models:

- Python PTM
- Java PTM
- All PTM

These 3 models are then fine-tuned on the downstream task of code2nl or Code
Summarization as defined in [CodeBERT](https://github.com/microsoft/CodeBERT).

All of the `.sh` scripts are defined to run the models as they were run for 
the thesis, and can be modified to produce different languages.

### Pre-Training

1. `sh setup.sh` - Sets up for pre-training using the [setup script](setup.sh). This runs a set of scripts that can be found under the [scripts directory](./scripts/). This sets up a virtual environment for python, installs all of the requirements, downloads the data, spreads it into the proper structure, then creates the appropriate tokenizers.

2. `sh pre-train.sh` - Run the [pre-training script](./pre_train/pre-train.sh). This will run 
    on the default values for one of the 3 models. These take a really long time (3 days at a
    minimum running on a GPU) so only 1 is generated in the script.

### Fine-Tuning

#### Code2NL
Look at the [code2nl README](./code2nl/README.md) for full steps.

## TLDR
At a really simple level RoBERTaCODE is a language model that can take a segment 
of code, and produce a summary of what the code is doing. For example, you have
found the following function in some code you are working on (from the [YouGet 
repository](https://github.com/soimort/you-get)):

```py
def get_video_url_from_video_id(video_id):
    # from js
    data = [""] * 256
    for index, _ in enumerate(data):
        t = index
        for i in range(8):
            t = -306674912 ^ unsigned_right_shitf(t, 1) if 1 & t else unsigned_right_shitf(t, 1)
        data[index] = t

    def tmp():
        rand_num = random.random()
        path = "/video/urls/v/1/toutiao/mp4/{video_id}?r={random_num}".format(video_id=video_id,
                                                                              random_num=str(rand_num)[2:])
        e = o = r = -1
        i, a = 0, len(path)
        while i < a:
            e = ord(path[i])
            i += 1
            if e < 128:
                r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ e)]
            else:
                if e < 2048:
                    r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (192 | e >> 6 & 31))]
                    r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | 63 & e))]
                else:
                    if 55296 <= e < 57344:
                        e = (1023 & e) + 64
                        i += 1
                        o = 1023 & t.url(i)
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (240 | e >> 8 & 7))]
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | e >> 2 & 63))]
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | o >> 6 & 15 | (3 & e) << 4))]
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | 63 & o))]
                    else:
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (224 | e >> 12 & 15))]
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | e >> 6 & 63))]
                        r = unsigned_right_shitf(r, 8) ^ data[255 & (r ^ (128 | 63 & e))]

        return "https://ib.365yg.com{path}&s={param}".format(path=path, param=unsigned_right_shitf(r ^ -1, 0))

    while 1:
        url = tmp()
        if url.split("=")[-1][0] != "-":
            return url
```
 
We don't have any additional context as, in our example, the original authors 
did not add any comments. After feeding this function into the model, it 
produces a text summary of the codes functionality:

```
Splicing URLs according to video ID to get video details
```

This is great, as we now have some more context as to what this function is 
trying to do!

To understand how effective this is, you can look at the original
function and the [comment that the author left describing the functionality](https://github.com/soimort/you-get/blob/b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/ixigua.py#L35).
**To save you the click, the model produced the exact comment produced by the author of the code.**

