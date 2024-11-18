import itertools
import random
from typing import Dict, List, Tuple

# a subset of scrabble words that are tokenized to single tokens with the scheme "cow2cat"
# There are 707 out of 1338 words remaining
scrabble_words_pass_number = [
    "cow",
    "orb",
    "biz",
    "hop",
    "pot",
    "sav",
    "sen",
    "apt",
    "att",
    "nos",
    "urb",
    "den",
    "pay",
    "fen",
    "mil",
    "ans",
    "div",
    "zen",
    "fat",
    "ami",
    "dal",
    "deb",
    "apo",
    "ens",
    "gon",
    "may",
    "ser",
    "och",
    "oud",
    "rut",
    "gas",
    "jaw",
    "ers",
    "rig",
    "ass",
    "flu",
    "hay",
    "hin",
    "del",
    "fas",
    "god",
    "pat",
    "jug",
    "fab",
    "cid",
    "een",
    "das",
    "and",
    "war",
    "man",
    "nil",
    "par",
    "nah",
    "pow",
    "bam",
    "bah",
    "lap",
    "git",
    "pod",
    "ick",
    "ord",
    "top",
    "elf",
    "arm",
    "tar",
    "his",
    "sip",
    "zip",
    "moz",
    "gen",
    "yum",
    "ups",
    "old",
    "rho",
    "pit",
    "sol",
    "urn",
    "sec",
    "pip",
    "leg",
    "ask",
    "lin",
    "lod",
    "sty",
    "box",
    "ilk",
    "imp",
    "bal",
    "gov",
    "pic",
    "reb",
    "mon",
    "sei",
    "lar",
    "tax",
    "bug",
    "app",
    "sad",
    "bad",
    "sez",
    "sup",
    "mis",
    "tee",
    "mos",
    "aid",
    "mac",
    "say",
    "alt",
    "nom",
    "bel",
    "toc",
    "pel",
    "rib",
    "bra",
    "fit",
    "kas",
    "ned",
    "fly",
    "fid",
    "sin",
    "obs",
    "lob",
    "awk",
    "che",
    "mix",
    "too",
    "ifs",
    "eve",
    "win",
    "tan",
    "sat",
    "vor",
    "tit",
    "wit",
    "dep",
    "ani",
    "elm",
    "hot",
    "sit",
    "two",
    "cal",
    "cap",
    "jab",
    "chi",
    "jud",
    "pan",
    "ill",
    "lit",
    "erm",
    "sus",
    "wig",
    "vas",
    "led",
    "its",
    "ply",
    "was",
    "pes",
    "ale",
    "raj",
    "lip",
    "fir",
    "kop",
    "ach",
    "caf",
    "art",
    "lat",
    "tet",
    "sex",
    "off",
    "cab",
    "cel",
    "get",
    "how",
    "lee",
    "lex",
    "pig",
    "sam",
    "sis",
    "has",
    "pol",
    "cot",
    "dag",
    "thy",
    "gan",
    "ary",
    "pub",
    "tom",
    "bob",
    "ess",
    "fig",
    "rep",
    "not",
    "dif",
    "eta",
    "fun",
    "ked",
    "nas",
    "sky",
    "fox",
    "ire",
    "lis",
    "gio",
    "ops",
    "egg",
    "err",
    "bro",
    "goo",
    "ren",
    "iso",
    "sig",
    "nut",
    "cob",
    "geo",
    "dam",
    "dig",
    "hit",
    "did",
    "lay",
    "sic",
    "the",
    "sub",
    "car",
    "cur",
    "ice",
    "ech",
    "hog",
    "lot",
    "per",
    "yes",
    "kir",
    "use",
    "bee",
    "aba",
    "die",
    "rip",
    "tot",
    "pix",
    "rad",
    "fin",
    "son",
    "bac",
    "inn",
    "fan",
    "bio",
    "bur",
    "tic",
    "bor",
    "emo",
    "lie",
    "tod",
    "rem",
    "rex",
    "aby",
    "bos",
    "new",
    "ben",
    "pir",
    "key",
    "ich",
    "try",
    "act",
    "awe",
    "mux",
    "bas",
    "rum",
    "med",
    "men",
    "sap",
    "fee",
    "pun",
    "tam",
    "ide",
    "rug",
    "ons",
    "max",
    "oxy",
    "vie",
    "oma",
    "dim",
    "als",
    "rin",
    "uts",
    "ern",
    "hum",
    "web",
    "ion",
    "end",
    "ten",
    "tas",
    "ped",
    "peg",
    "mot",
    "mol",
    "tip",
    "nam",
    "hat",
    "out",
    "day",
    "gar",
    "pad",
    "tid",
    "kid",
    "uns",
    "don",
    "erk",
    "cha",
    "lig",
    "oft",
    "ume",
    "tin",
    "tes",
    "yer",
    "wow",
    "van",
    "las",
    "spy",
    "dex",
    "mas",
    "gin",
    "jam",
    "mes",
    "ley",
    "sim",
    "tec",
    "sum",
    "ret",
    "kat",
    "het",
    "var",
    "soc",
    "mak",
    "are",
    "hyp",
    "bud",
    "org",
    "ins",
    "mus",
    "mag",
    "mob",
    "bag",
    "roc",
    "ode",
    "rap",
    "ard",
    "pre",
    "eco",
    "gid",
    "ort",
    "rod",
    "bib",
    "dad",
    "dog",
    "vim",
    "joy",
    "arb",
    "ama",
    "iff",
    "hic",
    "ere",
    "cut",
    "aff",
    "axe",
    "own",
    "ado",
    "hen",
    "let",
    "sel",
    "icy",
    "buy",
    "bes",
    "gay",
    "obe",
    "mir",
    "got",
    "fer",
    "bet",
    "deg",
    "meg",
    "sar",
    "bot",
    "kon",
    "ace",
    "dom",
    "pas",
    "rob",
    "dot",
    "est",
    "ate",
    "pro",
    "low",
    "til",
    "gap",
    "bit",
    "law",
    "fax",
    "ais",
    "gif",
    "row",
    "ant",
    "age",
    "fed",
    "pie",
    "syn",
    "lac",
    "ure",
    "boo",
    "doc",
    "ram",
    "way",
    "zag",
    "auf",
    "ios",
    "lei",
    "vis",
    "hom",
    "nor",
    "pen",
    "los",
    "lib",
    "ras",
    "pst",
    "sha",
    "tat",
    "met",
    "dee",
    "ken",
    "ana",
    "pal",
    "pac",
    "sur",
    "ran",
    "put",
    "ego",
    "ash",
    "eat",
    "cod",
    "lid",
    "cue",
    "nod",
    "lav",
    "mel",
    "sal",
    "ink",
    "beg",
    "rat",
    "vin",
    "any",
    "vid",
    "col",
    "who",
    "now",
    "tie",
    "bar",
    "som",
    "woo",
    "kin",
    "air",
    "toy",
    "due",
    "rew",
    "boy",
    "opt",
    "fix",
    "cad",
    "hoc",
    "ora",
    "six",
    "tap",
    "eng",
    "rom",
    "erg",
    "dry",
    "reg",
    "bus",
    "via",
    "sed",
    "bat",
    "ain",
    "aim",
    "all",
    "kor",
    "far",
    "lug",
    "rot",
    "dob",
    "ute",
    "him",
    "cep",
    "bru",
    "pur",
    "bon",
    "dis",
    "wan",
    "els",
    "mic",
    "raw",
    "vol",
    "hes",
    "our",
    "cam",
    "ket",
    "def",
    "nid",
    "ous",
    "tab",
    "neg",
    "wat",
    "bed",
    "eth",
    "rez",
    "run",
    "rag",
    "aka",
    "her",
    "lad",
    "nim",
    "tau",
    "rim",
    "see",
    "hos",
    "lev",
    "ban",
    "hex",
    "wyn",
    "tel",
    "vac",
    "pop",
    "elt",
    "jet",
    "lam",
    "rit",
    "jun",
    "ole",
    "tag",
    "fib",
    "net",
    "ann",
    "add",
    "ape",
    "owl",
    "ski",
    "tor",
    "non",
    "din",
    "ray",
    "vig",
    "cop",
    "poi",
    "ago",
    "fur",
    "nav",
    "big",
    "cos",
    "ing",
    "tog",
    "wis",
    "bum",
    "pec",
    "gun",
    "uni",
    "zap",
    "rec",
    "res",
    "won",
    "pee",
    "pus",
    "sun",
    "can",
    "bay",
    "eds",
    "mid",
    "tea",
    "map",
    "lab",
    "rub",
    "but",
    "ged",
    "yet",
    "fon",
    "nat",
    "eff",
    "ted",
    "amp",
    "dap",
    "mad",
    "ear",
    "ell",
    "mut",
    "mod",
    "rev",
    "hue",
    "mar",
    "qua",
    "seg",
    "had",
    "ref",
    "gee",
    "orc",
    "ids",
    "pos",
    "cor",
    "ave",
    "wed",
    "odd",
    "wen",
    "bin",
    "bis",
    "hid",
    "phi",
    "gal",
    "sea",
    "cry",
    "job",
    "vat",
    "abb",
    "bid",
    "gor",
    "jak",
    "hip",
    "fil",
    "few",
    "ger",
    "hem",
    "dit",
    "hey",
    "hub",
    "hon",
    "mor",
    "pis",
    "san",
    "ump",
    "eye",
    "nth",
    "lag",
    "nie",
    "dup",
    "alf",
    "arc",
    "yaw",
    "gel",
    "han",
    "log",
    "cis",
    "pet",
    "cum",
    "era",
    "kos",
    "ore",
    "rif",
    "red",
    "pin",
    "mem",
    "you",
    "bow",
    "lor",
    "nap",
    "bye",
    "why",
    "ark",
    "foo",
    "owe",
    "she",
    "mal",
    "abs",
    "hap",
    "nan",
    "set",
    "oil",
    "mun",
    "ail",
    "cat",
    "for",
    "rid",
    "zig",
    "jar",
    "dos",
    "nip",
    "ads",
    "ava",
    "dan",
    "kit",
    "hoe",
    "one",
    "nit",
    "cup",
    "ton",
    "fra",
    "ham",
    "psi",
    "vet",
    "tex",
    "mat",
    "spa",
    "nob",
    "gem",
    "lah",
    "mom",
    "gam",
    "pak",
    "vox",
    "con",
    "lux",
    "rud",
    "yen",
    "cit",
    "dev",
    "boa",
]
letter_chars = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
all_2tuples = ["".join(t) for t in itertools.product(letter_chars, repeat=2)]


def solve_pen(s):
    s = (
        s.replace("answer:", "").split(":")[-1].strip()
    )  # safety measure, if the input has a task encoding and includes "answer:"
    words = s.split(" ")
    if (
        len(words[0]) == 3
    ):  # this is in case the triplet transformation has been applied
        l = 3
    else:
        l = 2
    answer = []
    current_index = 0
    while current_index is not None:
        answer.append(words[current_index + 1])
        next_index = None
        for index, word in enumerate(words[2:], start=2):
            if word.startswith(words[current_index][-l:]):
                next_index = index
        current_index = next_index
    return " ".join(answer)


def get_green_matching_words(pen_x_words, pen_y_words):
    l = len(pen_x_words[0])  # is 2, and is 3 if we have the triplet tokentransform
    green_words = [pen_x_words[0]]
    for neigh in pen_y_words[1:]:
        for j in range(len(pen_x_words)):
            if neigh == pen_x_words[j]:
                green_words.append(pen_x_words[j - 1])  # get the left neighbour
                assert (
                    green_words[-2][-l:] == green_words[-1][:l]
                ), f"Error: the green words are no matching sequence: {green_words}"
    assert len(green_words) == len(
        pen_y_words
    ), f"Error: the green words are no matching sequence: {green_words}"
    return green_words


def transform_sample_to_triplets(pen_x_words, pen_y_words):
    word_beginnigs = [word[:2] for word in pen_x_words]
    word_endings = [word[-2:] for word in pen_x_words]
    identifiers = set(word_beginnigs) | set(word_endings)
    # make a random mapping to scrabble words
    random_scrabble_words = random.sample(scrabble_words_pass_number, len(identifiers))
    mapping = dict(zip(identifiers, random_scrabble_words))
    # apply the mapping
    x_words = [mapping[pen_x_words[0]], mapping[pen_x_words[1]]] + [
        mapping[word[:2]] + word[2:-2] + mapping[word[-2:]] for word in pen_x_words[2:]
    ]
    # ATTENTION: HERE WE HAVE SEQNEXT
    y_words = [mapping[pen_y_words[0]]] + [
        mapping[word[:2]] + word[2:-2] + mapping[word[-2:]] for word in pen_y_words[1:]
    ]
    return x_words, y_words


def remove_doublegangers(pen_x_words, pen_y_words):
    assert (
        len(pen_x_words[0]) == 2
    ), "This function must be called before triplet transformations"
    word_beginnigs = [word[:2] for word in pen_x_words]
    word_endings = [word[-2:] for word in pen_x_words]
    identifiers = set(word_beginnigs) | set(word_endings)

    random_more_2tuples = random.sample(all_2tuples, 200)
    random_more_2tuples = [t for t in random_more_2tuples if t not in identifiers]
    tuple_index = 0
    for neigh in pen_y_words[1:]:  # skip the first word as it has no doubleganger
        for j in range(len(pen_x_words)):
            if (
                neigh[:2] == pen_x_words[j][:2]
                and neigh[-2:] == pen_x_words[j][-2:]
                and neigh != pen_x_words[j]
            ):
                pen_x_words[j] = (
                    random_more_2tuples[tuple_index]
                    + pen_x_words[j][2:-2]
                    + random_more_2tuples[tuple_index + 1]
                )
                tuple_index += 2
    return pen_x_words, pen_y_words


def build_prompt(sample_list: List[Tuple[str, str]], config: Dict):
    for i in range(len(sample_list)):
        # preprocess to lists of words
        pen_x, pen_y = sample_list[i]
        pen_x = pen_x.replace("answer:", "").strip()
        pen_x = pen_x.split(":")[-1].strip()  # remove task identifier
        pen_x = pen_x.split(" ")
        pen_y = pen_y.split(" ")
        sample_list[i] = (pen_x, pen_y)

    if config["remove_doublegangers"]:
        sample_list = [remove_doublegangers(x, y) for x, y in sample_list]

    tokentransform = config["use_single_token_triplets"]
    if tokentransform:
        sample_list = [transform_sample_to_triplets(x, y) for x, y in sample_list]

    samples_with_green = [
        (x, y, get_green_matching_words(x, y)) for x, y in sample_list
    ]

    # description
    if config["has_description"]:
        description = f"""I give you a sequence of words. Each word has four characters plus a middle, words are separated by spaces. Start with the leftmost word. Output its neighbour. 
Then, match the last {'three' if tokentransform else 'two'} characters of the current word (i.e. not the neighbour) to the word starting with those {'three' if tokentransform else 'two'} characters. Again, output the neighbour. Do this until your current word (not the neighbour) has no match anymore.\n\n"""
    else:
        description = ""

    # COT
    if config["ask_for_cot"] == True:
        cot_question = """\nReason step by step."""
    elif config["ask_for_cot"] == "with_code":
        cot_question = """\nReason step by step. Then, use the code interpreter to solve the task."""
    else:
        cot_question = ""

    # few-shot examples
    examples = []
    if config["manyshot"]:
        assert not config["sge"], "Manyshot and SGE are not compatible"
        n_demonstrations = 32
    else:
        if config["ask_for_cot"] == "with_code":
            n_demonstrations = 1
        elif config["sge"]:
            n_demonstrations = 0
        else:
            n_demonstrations = 8
    assert (
        len(sample_list) > n_demonstrations
    ), f"len(sample_subset): {len(sample_list)}, n_samples: {n_demonstrations}"

    # process the few-shot examples
    l = 3 if tokentransform else 2
    fewshot_cot = config["fewshot_cot"]
    for pen_x, neighbour_sequence, matching_sequence in samples_with_green[
        0:n_demonstrations
    ]:
        new_example = f'Example: {" ".join(pen_x)}\n'
        if fewshot_cot == "2stages":
            new_example += f"First we find the matching sequence of words, then we find the neighbours.\n"
            new_example += f'The first word is "{matching_sequence[0]}".\n'
            for i in range(1, len(matching_sequence)):
                new_example += f'Now we need to find a word that starts with "{matching_sequence[i-1][-l:]}". The word is "{matching_sequence[i]}".\n'
            new_example += f'There is no word that starts with "{matching_sequence[-1][-l:]}", so we are done with the matching.\n'
            new_example += f"Now we need to find the neighbours of the matched words.\n"
            for i in range(len(neighbour_sequence)):
                new_example += f'The right neighbour of "{matching_sequence[i]}" is "{neighbour_sequence[i]}".\n'
            new_example += (
                f'Therefore the answer is: "' + " ".join(neighbour_sequence) + '"\n\n'
            )
        elif fewshot_cot:
            new_example += f'The leftmost word is "{matching_sequence[0]}". Its right neighbour is "{neighbour_sequence[0]}", so the first output word is "{neighbour_sequence[0]}".\n'
            for i in range(1, len(matching_sequence)):
                new_example += f'Now, we need to find a word that starts with "{matching_sequence[i-1][-l:]}". The word is "{matching_sequence[i]}". Its right neighbour is "{neighbour_sequence[i]}", so the next output word is "{neighbour_sequence[i]}".\n'
            new_example += f'There is no word that starts with "{matching_sequence[-1][-l:]}", so we are done with the matching.\n'
            new_example += (
                f'Therefore the answer is: "' + " ".join(neighbour_sequence) + '"\n\n'
            )
        else:
            new_example += f'Answer: {" ".join(neighbour_sequence)}\n\n'

        predicted_answer = solve_pen(" ".join(pen_x))
        assert predicted_answer == " ".join(
            neighbour_sequence
        ), f'predicted_answer: {predicted_answer}, sample["answer"]: {neighbour_sequence}'
        examples.append(new_example)

    actual_pen_x, actual_pen_y, actual_green_words = samples_with_green[-1]

    if not config["sge"]:
        final_task = f'Your question: {" ".join(actual_pen_x)}'
        prompt = (
            f"""{description}"""
            + "".join(examples)
            + final_task
            + f"""{cot_question}\nClearly mark your answer by writing 'Answer: <your answer>' as last line."""
        )
    else:
        global SGE_PROMPT
        insertion = (
            f"""{description}"""
            + "Sequence: "
            + " ".join(actual_pen_x)
            + f"""{cot_question}"""
        )
        prompt = SGE_PROMPT.replace("[PROBLEM DESCRIPTION]", insertion)

    true_answer = " ".join(actual_pen_y)

    return prompt, true_answer


SGE_PROMPT = """
Your task is to tackle algorithmic problems. When presented with an algorithmic problem, recall relevant problems as examples. Afterward, proceed to solve the initial problem. 

# Problem: [PROBLEM DESCRIPTION]

# Instructions: 
## Relevant Problems: 
Recall three examples of algorithmic problems that are relevant to the initial problem. Your problems should be distinct from each other and from the initial problem (e.g., involving different numbers and names and instructions). For each problem:
 - After "Q: ", describe the problem
 - After "A: ", explain the solution and enclose the ultimate answer in \boxed{}. 

## Solve the Initial Problem: 
Q: Copy and paste the initial problem here. 
A: Explain the solution and enclose the ultimate answer in \boxed{} here. Make sure the last line is \boxed{YOUR ANSWER}.
""".strip()
