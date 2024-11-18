import itertools
from typing import Dict, List, Tuple

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


def solve_perm(s):
    s = (
        s.replace("answer:", "").split(":")[-1].strip()
    )  # safety measure, if the input has a task encoding and includes "answer:"
    words = s.split("|")[0].strip().split(" ")
    start = s.split("|")[1].strip()
    answer_next = []
    matching_seq = []
    current_word = start
    idx = words.index(current_word)
    n_left = 0
    answer_n_left = []
    while True:
        matching_seq.append(current_word)
        answer_next.append(words[idx + 1])
        answer_n_left.append(n_left)
        next_word = [
            (w, i) for i, w in enumerate(words) if w.startswith(current_word[-2:])
        ]
        if len(next_word) == 0:
            break
        assert len(next_word) == 1
        current_word, new_idx = next_word[0]
        if new_idx < idx:
            n_left += 1
        idx = new_idx
        if current_word in matching_seq:
            break
    # reverse
    matching_seq = matching_seq[::-1]
    answer_n_left = answer_n_left[::-1]
    answer_c = list(range(len(answer_n_left) - 1, -1, -1))
    return matching_seq, answer_n_left, answer_c


def build_prompt(sample_list: List[Tuple[str, str]], config: Dict):
    for i in range(len(sample_list)):
        # preprocess to list of words
        perm_x, perm_y = sample_list[i]
        perm_x = perm_x.replace("answer:", "").strip()
        perm_x = perm_x.split(":")[-1].strip()
        perm_x_words = perm_x.split("|")[0].strip()
        perm_x_start = perm_x.split("|")[1].strip()
        match_seq_rev, n_lefts_rev, counts_rev = solve_perm(perm_x)
        perm_x_words = perm_x_words.split(" ")
        perm_y = perm_y.split(" ")
        assert all(
            [
                f"{word}.{n_left*count}" == y
                for word, n_left, count, y in zip(
                    match_seq_rev, n_lefts_rev, counts_rev, perm_y
                )
            ]
        )
        sample_list[i] = (
            perm_x,
            perm_y,
            perm_x_words,
            perm_x_start,
            match_seq_rev,
            n_lefts_rev,
            counts_rev,
        )

    # description
    if config["has_description"]:
        description = """I give you a sequence of words. The last work (after the "|") is the word to start with. Now match match the last two characters of the current word to the word starting with those two characters. If this match was going to the left, i.e. the matched word is left of the current word in the sequence, increase a variable counting the number of left matchings. Do this until your current word has no match anymore. 
Finally, output this sequence of words, in reverse order in the format word.x where x is the number of left matchings until the output word times the number of matchings until the output word. Example answer: abcd.4 efab.1 ghef.0\n\n"""
    else:
        description = ""
    if config["manyshot"] == True:
        cot_question = """\nReason step by step."""

    # COT
    elif config["ask_for_cot"] == "with_code":
        cot_question = """\nReason step by step. Then, use the code interpreter to solve the task."""
    else:
        cot_question = ""

    # few-shot examples
    examples = []
    if config["manyshot"]:
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
    fewshot_cot = config["fewshot_cot"]
    assert fewshot_cot in [True, False, "enumerate"]
    for (
        perm_x,
        perm_y,
        perm_x_words,
        perm_x_start,
        match_seq_rev,
        n_lefts_rev,
        counts_rev,
    ) in sample_list[0:n_demonstrations]:
        new_example = f"\n\nExample: {perm_x}\n"
        if fewshot_cot == True or fewshot_cot == "enumerate":
            current_word = perm_x_start
            if fewshot_cot == "enumerate":
                new_example += f"First, let's enumerate the words:\n"
                for i in range(len(perm_x_words)):
                    new_example += f"{i+1}:{perm_x_words[i]}\n"
                new_example += "\n"
            new_example += (
                f'Starting with "{perm_x_start}", let\'s match and calculate:\n\n'
            )
            matching_seq = match_seq_rev[
                ::-1
            ]  # the true answer has the matching sequence reversed, so we reverse again, to get it forward
            n_lefts = n_lefts_rev[::-1]
            for i in range(1, len(matching_seq)):
                next_match = matching_seq[i]
                idx_current = perm_x_words.index(current_word)
                idx_next = perm_x_words.index(next_match)
                is_left_match = n_lefts[i] > n_lefts[i - 1]
                assert is_left_match == (idx_next < idx_current)
                n_left_next = n_lefts[i]

                new_example += f'"{current_word}" matches with "{next_match}". '
                if is_left_match:
                    if fewshot_cot == "enumerate":
                        new_example += f'The word "{current_word}" is {idx_current+1}th and "{next_match}" is {idx_next+1}th, so {n_left_next} left matches so far.\n'
                    else:
                        new_example += f"This is a left match, {n_left_next} left matches so far.\n"
                else:
                    if fewshot_cot == "enumerate":
                        new_example += f'The word "{current_word}" is {idx_current+1}th and "{next_match}" is {idx_next+1}th, so {n_left_next} left matches so far.\n'
                    else:
                        new_example += (
                            f"No left match, {n_left_next} left matches so far.\n"
                        )
                current_word = next_match
            new_example += f'\nThere are no further matches for "{current_word}", so we end the sequence here.'
            if fewshot_cot == "calculate" or fewshot_cot == "enumerate":
                new_example += f"\n\nFinally, we calculate the number of left matches times the number of matches for each word and get:\n\n"
                for i in range(len(matching_seq)):
                    word = matching_seq[i]
                    n_left = n_lefts[i]
                    new_example += f"{word}: {n_left}*{i}={n_left*i}\n"
            else:
                new_example += f"\n\nReversing the sequence and formatting it as per the instructions, we get:\n\n"
                for i in range(len(match_seq_rev)):
                    new_example += (
                        f"{match_seq_rev[i]}.{n_lefts_rev[i]*(len(n_lefts_rev)-i-1)}\n"
                    )
            new_example += f'\nThus, the answer is: "' + " ".join(perm_y) + '".'
        else:
            new_example += f'Answer: {" ".join(perm_y)}\n\n'
        predicted_answer = " ".join(
            [
                match_seq_rev[i]
                + "."
                + str(n_lefts_rev[i] * (len(match_seq_rev) - i - 1))
                for i in range(len(match_seq_rev))
            ]
        )
        assert predicted_answer == " ".join(perm_y)
        examples.append(new_example)

    actual_perm_x, actual_perm_y, _, _, _, _, _ = sample_list[-1]
    if not config["sge"]:
        final_task = f"\n\nYour question: {actual_perm_x}"
        prompt = (
            f"""{description}"""
            + "".join(examples)
            + final_task
            + f"""{cot_question}\nClearly mark your answer by writing 'Answer: <your answer>' as last line."""
        )
    else:
        prompt_template = open("SGE_prompt.txt", "r").read()
        insertion = (
            f"""{description}""" + "Sequence: " + actual_perm_x + f"""{cot_question}"""
        )
        prompt = prompt_template.replace("[PROBLEM DESCRIPTION]", insertion)

    true_answer = " ".join(actual_perm_y)

    return prompt, true_answer
