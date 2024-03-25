import re
import nltk
import json
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer


PRONOUN = dict(
    {
        "where": "1",
        "what": "2",
        "which": "3",
        "when": "4",
        "how": "5",
        "whom": "6",
        "whose": "6",
        "why": "7",
    }
)
SP_RELATION = set(
    [
        "near",
        "top",
        "front",
        "south",
        "west",
        "north",
        "east",
        "southwest",
        "southeast",
        "northwest",
        "northeast",
        "part",
        "seat",
        "center",
    ]
)
POUNC = set(
    [
        ".",
        ",",
        "'",
        "?",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "*",
        "(",
        ")",
        "[",
        "]",
        ";",
        '"',
    ]
)
NOUN_TAG = set(["NN", "NNS", "NNP", "NNPS"])
ADJ_TAG = set(["JJ", "JJR", "JJS"])
NO = set(
    [
        "a",
        "an",
        "the",
        "and",
        "do",
        "did",
        "does",
        "be",
        "am",
        "is",
        "was",
        "are",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    ]
)
VERB_TAG = set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
PREP_TAG = set(["IN", "TO"])


# load gazetteer in dictionary
def load_gazetteer(fga):
    fga = open(fga, "r")
    gazetteer = dict()
    for line in fga.readlines():
        line = line.strip()
        gazetteer[line.lower()] = line
    fga.close()
    return gazetteer


# load word
def load_word(fword):
    words = set()
    fword = open(fword, "r")
    for line in fword.readlines():
        word = line.strip()
        words.add(word)
    fword.close()
    return words


# load place type
def load_pt(fpt):
    pt_set = set()
    pt_dict = dict()
    fpt = open(fpt, "r")
    for line in fpt.readlines():
        word = line.strip().split()
        pt_set.add(word[1])
        pt_dict[word[0]] = word[1]
    fpt.close()
    return pt_set, pt_dict


def load_data(fdata):
    fdata = open(fdata, "r")
    data = json.load(fdata)
    return data


def load_abbr(fabbr):
    abbr = dict()
    fabbr = open(fabbr, "r")
    for line in fabbr.readlines():
        line = line.strip().split()
        abbr[line[-1]] = " ".join(line[:-1])
    return abbr


# filter from dict 'ga' delete set 'topn'
def filterr(ga, topn):
    for word in topn:
        if word in ga:
            ga.pop(word)
    return ga


def process_data(data, ga, abbr, pt_set, pt_dict, actv, stav, sp_prep):
    for item in tqdm(data):
        # process query
        query = item["query"].lower()
        analyze = process_sentence(
            query, ga, abbr, pt_set, pt_dict, actv, stav, sp_prep
        )
        item["queryAnalyze"] = analyze
        # process answer
        if "answers" in item:
            item["answersAnalyze"] = list()
            for answer in item["answers"]:
                analyze = process_sentence(
                    answer.lower(), ga, abbr, pt_set, pt_dict, actv, stav, sp_prep
                )
                item["answersAnalyze"].append(analyze)
    return data


def process_sentence(query, ga, abbr, pt_set, pt_dict, actv, stav, sp_prep):

    # tokenization
    tokens = nltk.word_tokenize(query)

    # delete punctuation mark
    tokens = list(filter(lambda x: x not in POUNC, tokens))

    # expand abbreviation
    tokens = [abbr[token] if (token in abbr) else token for token in tokens]
    tokens = " ".join(tokens).split(" ")

    # tagging
    query_len = len(tokens)
    code = ["." for i in range(query_len)]
    if query_len == 1:
        tokens = ["<IGNORE>"]
    tag = nltk.pos_tag(tokens)

    # place name (greedy match)
    i = 0
    pname = []
    while i < query_len:
        for j in range(query_len, i, -1):
            phrase = " ".join(tokens[i:j])
            if phrase in ga:
                tokens[i:j] = ga[phrase].split()
                code[i] = "n"
                code[i + 1 : j] = ["-" for k in range(j - i - 1)]  # noqa: E203
                pname.append(ga[phrase])
                i = j
                break
        i += 1

    for i, token in enumerate(tokens):
        # filter
        if token in NO:
            code[i] = ","
        # normalize place type
        if token in pt_dict:
            tokens[i] = pt_dict[token]
    sen = " ".join(tokens)

    ptype = []
    activity = []
    stative = []
    sp_relation = []

    for i, token in enumerate(tokens):
        if code[i] == ".":
            # pronoun
            if token in PRONOUN:
                code[i] = PRONOUN[token]
            # place type
            elif token in pt_set:
                code[i] = "t"
                ptype.append(token)
            # other object
            elif tag[i][1] in NOUN_TAG:
                code[i] = "o"
            # verb (simple dict lookup)
            elif tag[i][1] in VERB_TAG:
                stem = WordNetLemmatizer().lemmatize(tag[i][0], "v")
                if stem in actv:
                    code[i] = "a"
                    activity.append(stem)
                elif stem in stav:
                    code[i] = "s"
                    stative.append(stem)
            # preposition
            elif (
                (tag[i][1] in PREP_TAG)
                and (token in sp_prep)
                and (i + 1 < query_len)
                and (code[i + 1] == "n")
            ):
                code[i] = "r"
                spr = token + " " + tokens[i + 1]
                k = i + 2
                while k < query_len and code[k] == "-":
                    spr = spr + " " + tokens[k]
                    k += 1
                sp_relation.append(spr)

    quality = []
    # quality
    for i, token in enumerate(tokens):
        # place quality
        if (
            (code[i] == ".")
            and (tag[i][1] in ADJ_TAG)
            and (i + 1 < query_len)
            and (code[i + 1] == "t")
        ):
            code[i] = "q"
            quality.append(token)

    code = re.sub("[\.,-]", "", ("".join(code)))  # noqa: W605

    analyze = dict()
    analyze["placeName"] = pname
    analyze["placeType"] = ptype
    analyze["code"] = code
    analyze["sentence"] = sen
    analyze["quality"] = quality
    analyze["activity"] = activity
    analyze["stative"] = stative
    analyze["sp_relation"] = sp_relation
    return analyze


def output(data, foutput):
    with open(foutput, "w") as f:
        json.dump(data, f, indent=4)


def main():

    fdata = "data/raw_data/dataset-v2.1-location-queries.json"
    fga = "data/gazetteer/gazetteer.txt"
    fabbr = "data/gazetteer/abbr.txt"
    fcommon_word = "data/common_word/top10000.txt"
    fpt = "data/place_type/place_type.txt"
    factv = "data/verb/action_verb.txt"
    fstav = "data/verb/stative_verb.txt"
    foutput = "data/result/result.json"
    fsp_prep = "data/prep/prep.txt"

    print("loading data ... ")
    pt_set, pt_dict = load_pt(fpt)
    actv = load_word(factv)
    stav = load_word(fstav)
    sp_prep = load_word(fsp_prep)
    data = load_data(fdata)
    abbr = load_abbr(fabbr)
    ga = load_gazetteer(fga)

    topn = load_word(fcommon_word)
    ga = filterr(ga, topn)
    print("done")

    print("start processing ...")
    result = process_data(data, ga, abbr, pt_set, pt_dict, actv, stav, sp_prep)
    print("done")

    print("writing file ...")
    output(result, foutput)
    print("done")


main()
