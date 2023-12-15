import csv
from pathlib import Path
from nltk import ngrams
from collections import defaultdict
from random import choices
import json


MESSAGES = Path("package") / "messages"
MODEL = Path(".") / "markov.json"
LENGTH_NGRAM = 7
START_NGRAM = "\0" * LENGTH_NGRAM
END_NGRAM = "\r" * LENGTH_NGRAM


def yield_text():
    for filename in MESSAGES.glob("*/*.csv"):
        with open(filename) as f:
            next(f)  # Ignore header
            for id_, timestamp, contents, attachments in csv.reader(f):
                if contents:
                    yield contents


def get_ngrams(text):
    n = ["".join(n)
         for n in ngrams(f"{START_NGRAM}{text}{END_NGRAM}", LENGTH_NGRAM)]
    return ngrams(n, 2)


def train():
    print("Model missing, training...")
    markov = defaultdict(dict)

    for index, text in enumerate(yield_text()):
        for previous, current in get_ngrams(text):
            markov[previous][current] = markov[previous].get(current, 0) + 1
        if not index % 10000:
            print(index)

    with open(MODEL, "w") as f:
        json.dump(markov, f, indent=2)


def infer():
    MAX_LENGTH = 500
    ngram = START_NGRAM
    i = 0
    while ngram != END_NGRAM:
        i += 1
        if i == MAX_LENGTH:
            break

        options = markov[ngram]
        # ngram = choices(list(options.keys()),
        #                 list(options.values()))[0]
        ngram = choices(list(options.keys()),
                        list([i**1.2 for i in options.values()]))[0]
        # ngram = max(options, key=options.get)
        yield ngram[-1]


if __name__ == "__main__":
    if not MODEL.exists():
        train()

    with open(MODEL) as f:
        markov = json.load(f)

    while not input():
        print("".join(infer()))
        print()
