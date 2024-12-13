import itertools

import numpy as np

letter_chars = list("abcdefghijklmnopqrstuvwxyz")
big_letter_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
number_chars = list("0123456789")


class PointerExecutionNeighbour:
    def __init__(self, min_len, max_len, min_hops, max_hops, sub_task="next"):
        # import pdb; pdb.set_trace()
        self.length_low = min_len  # min length of the sequence
        assert (
            self.length_low >= 2
        ), "Configuration Error: Minimum length must be at least 2"
        self.length_high = max_len + 1  # max length of the sequence
        self.hops_low = min_hops  # min length of the matching sequence
        self.hops_higher = max_hops + 1  # max length of the matching sequence
        self.task = sub_task  # composition or subtask identifier
        assert self.task in ["seq", "next", "seqnext"]
        self.all_2tuples = [
            "".join(t) for t in itertools.product(letter_chars, repeat=2)
        ]
        self.data_choices = list(
            number_chars[:8]
        )  # the choices for the middle character of "words"

    def generate_double_pointer_execution(self, n_samples):
        lengths = np.arange(self.length_low, self.length_high)
        samples = []
        answers = []
        # import pdb; pdb.set_trace()
        while len(samples) < n_samples:
            length = np.random.choice(lengths)
            n_matching_hops = np.random.choice(
                np.arange(self.hops_low, min(self.hops_higher, length // 2))
            )
            tuple_choices = np.random.choice(
                self.all_2tuples, length * 7, replace=False
            )
            # select the positions where the green maatching sequence will be
            positions = np.random.choice(
                np.arange(1, length), size=n_matching_hops, replace=False
            )
            cnt = 0
            question_words1 = ["" for _ in range(length)]
            question_words2 = ["" for _ in range(length)]
            remaining_positions = np.random.permutation(
                [i for i in range(1, length) if i not in positions]
            )
            question_words1[0] = tuple_choices[cnt]
            answer_learnseq = [question_words1[0]]
            for pos in positions:
                question_words1[pos] = (
                    tuple_choices[cnt]
                    + np.random.choice(self.data_choices)
                    + tuple_choices[cnt + 1]
                )
                answer_learnseq.append(question_words1[pos])
                cnt += 1
            cnt += 1
            cnt_confuse = cnt + length
            positions_next = np.random.permutation(positions)
            question_words2[0] = tuple_choices[cnt]
            answer = [question_words2[0]]
            # select the positions where the doppelgangers of the neighbours will be
            positions_confuse = np.setdiff1d(np.arange(1, length), positions_next)[
                0 : len(positions_next)
            ]
            np.random.shuffle(positions_confuse)
            for i, pos in enumerate(positions_next):
                two_big_letters = np.random.choice(
                    self.data_choices, size=2, replace=False
                )
                question_words2[pos] = (
                    tuple_choices[cnt] + two_big_letters[0] + tuple_choices[cnt + 1]
                )
                question_words2[positions_confuse[i]] = (
                    tuple_choices[cnt] + two_big_letters[1] + tuple_choices[cnt + 1]
                )
                answer.append(question_words2[pos])
                cnt += 1
                cnt_confuse += 1
            cnt = max(cnt, cnt_confuse) + 1
            remaining_next_positions = np.random.permutation(
                [
                    i
                    for i in range(1, length)
                    if i not in positions_next and i not in positions_confuse
                ]
            )
            for pos in remaining_positions:
                question_words1[pos] = (
                    tuple_choices[cnt]
                    + np.random.choice(self.data_choices)
                    + tuple_choices[cnt + 1]
                )
                cnt += 1
            cnt += 1
            for pos in remaining_next_positions:
                question_words2[pos] = (
                    tuple_choices[cnt]
                    + np.random.choice(self.data_choices)
                    + tuple_choices[cnt + 1]
                )
                cnt += 2
            answer_learnnext = [question_words2[0]]
            for pos in positions:
                answer_learnnext.append(question_words2[pos])
            answer_seqnext = []
            for i in range(len(answer_learnseq)):
                answer_seqnext.append(answer_learnseq[i])
                answer_seqnext.append(answer_learnnext[i])
            answer.reverse()
            question_words = []
            for i in range(length):
                question_words.append(question_words1[i])
                question_words.append(question_words2[i])
            question_str = (
                f"pe {self.task}: "
                + " ".join(["".join(x) for x in question_words])
                + " answer: "
            )
            samples.append(question_str)
            if self.task == "seq":
                answers.append(" ".join(answer_learnseq))
            elif self.task == "seqnext":
                answers.append(" ".join(answer_seqnext))
            elif self.task == "next":
                answers.append(" ".join(answer_learnnext))
        return samples, answers

    def generate(self, n_samples):
        samples, answers = self.generate_double_pointer_execution(n_samples)
        samples = [(x, y) for x, y in zip(samples, answers)]
        return samples


class PointerExecutionReverseMulticount:
    def __init__(self, min_len, max_len, sub_task="seq"):
        self.length_low = min_len  # min length of the sequence
        self.length_higher = max_len + 1  # max length of the sequence
        self.task = sub_task  # composition or subtask identifier
        assert self.task in ["seq", "multiseq", "seqrev", "multiseqrev"]
        self.all_2tuples = [
            "".join(t) for t in itertools.product(letter_chars, repeat=2)
        ]

    def generate_samples(self, n_samples):
        lengths = np.arange(self.length_low, self.length_higher)
        samples = []
        answers = []
        for _ in range(n_samples):
            length = np.random.choice(lengths)
            tuple_choices = np.random.choice(
                self.all_2tuples, length + 3, replace=False
            )
            last_word = tuple_choices[-3] + tuple_choices[-2]
            shuffled_tuple_choices1 = np.random.permutation(tuple_choices[:-3])
            shuffled_tuple_choices2 = np.random.permutation(tuple_choices[:-3])
            words = [
                ch1 + ch2
                for ch1, ch2 in zip(shuffled_tuple_choices1, shuffled_tuple_choices2)
            ]
            start = np.random.choice(words)
            words.append(last_word)
            if "rev" not in self.task:
                answer = self.solve_seqnext(words, start, self.task)
            else:
                # change the 2tuple of the start of the start word to a random one
                idx = words.index(start)
                words[idx] = tuple_choices[-1] + words[idx][2:]
                start = words[idx]
                answer, answer_n_left = self.solve_seqnext(words, start, self.task)
                if self.task == "seqrev":
                    answer = reversed([f"{w}" for i, w in enumerate(answer)])
                if self.task == "multiseqrev":
                    answer = reversed(
                        [
                            f"{w}.{i*n}"
                            for i, (w, n) in enumerate(zip(answer, answer_n_left))
                        ]
                    )
            question = (
                f"prand {self.task}: " + " ".join(words) + " | " + start + " answer: "
            )
            samples.append(question)
            answers.append(" ".join(answer))
        return samples, answers

    def solve_seqnext(self, words, start, mode):
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
            if len(next_word) == 0 and "rev" in mode:
                break
            assert len(next_word) == 1
            current_word, new_idx = next_word[0]
            if new_idx < idx:
                n_left += 1
            idx = new_idx
            if current_word in matching_seq:
                break
        if "rev" in mode:
            return matching_seq, answer_n_left
        if "multi" in mode:
            answer = []
            for i, (w, n) in enumerate(zip(matching_seq, answer_n_left)):
                answer.append(f"{w}.{i*n}")
            return answer
        return matching_seq

    def generate(self, n_samples):
        samples, answers = self.generate_samples(n_samples)
        samples = [(x, y) for x, y in zip(samples, answers)]
        return samples


class HighestSubseqenceNoNeighboursTask:
    def __init__(self, min_len, max_len, sub_task="sum"):
        self.length_low = min_len  # min length of the sequence in numbers
        self.max_len = max_len  # max length of the sequence in numbers, inclusive
        self.task = sub_task
        assert self.task in ["sum", "exec"]

    def highest_subseq_no_neighbours(self, numbers):
        highest_until_i = [(-1, -1, "n") for _ in range(len(numbers))]
        highest_until_i[0] = (max(numbers[0], 0), -1, "y" if numbers[0] > 0 else "n")
        if numbers[1] > numbers[0]:
            highest_until_i[1] = (
                max(numbers[1], 0),
                -1,
                "y" if numbers[1] > 0 else "n",
            )
        else:
            highest_until_i[1] = (highest_until_i[0][0], 0, "n")
        for i in range(2, len(numbers)):
            if highest_until_i[i - 1][0] > highest_until_i[i - 2][0] + numbers[i]:
                highest_until_i[i] = (highest_until_i[i - 1][0], i - 1, "n")
            else:
                highest_until_i[i] = (
                    highest_until_i[i - 2][0] + numbers[i],
                    i - 2,
                    "y" if numbers[i] > 0 else "n",
                )
        score = highest_until_i[-1][0]
        highest_subseq = []
        i = len(numbers) - 1
        while i >= 0:
            if highest_until_i[i][2] == "y":
                highest_subseq.append(numbers[i])
            i = highest_until_i[i][1]
        highest_subseq.reverse()
        execution_trace = "".join([x[2] + str(x[0]) for x in highest_until_i])
        return highest_subseq, score, execution_trace

    def generate_highest_subseq_no_neighbours(self, n_samples):
        lengths = np.arange(self.length_low, self.max_len + 1)
        samples = []
        for _ in range(n_samples):
            length = np.random.choice(lengths)
            numbers = np.random.randint(0, 10, size=length)
            (
                highest_subseq,
                answer_number,
                exec_trace,
            ) = self.highest_subseq_no_neighbours(numbers)
            if self.task == "sum":
                answer = str(answer_number)
            else:
                answer = exec_trace
            samples.append((numbers, answer))
        return samples

    def generate(self, n_samples):
        samples = self.generate_highest_subseq_no_neighbours(n_samples)
        samples = [
            (
                f"hsnn {self.task}: "
                + " ".join([str(x) for x in numbers])
                + " answer: ",
                answer,
            )
            for numbers, answer in samples
        ]
        return samples


class MultiplicationTask:
    def __init__(self, max_len_left=6, max_len_right=6, sub_task="mul"):
        self.length_higher_left = max_len_left + 1  # max length of the left number
        self.length_higher_right = max_len_right + 1  # max length of the right number
        self.task = sub_task  # composition or subtask identifier
        assert self.task in ["sum", "mul"]

    def generate_calculate_random(self, n_samples):
        lengths_left = np.arange(1, self.length_higher_left)
        lengths_right = np.arange(1, self.length_higher_right)
        samples = []
        for _ in range(n_samples):
            length_left = np.random.choice(lengths_left)
            length_right = np.random.choice(lengths_right)
            left = np.random.randint(0, 10, size=length_left)
            right = np.random.randint(0, 10, size=length_right)
            left_number = int("".join([str(x) for x in left]))
            right_number = int("".join([str(x) for x in right]))
            if self.task == "sum":
                answer = left_number + right_number
            elif self.task == "mul":
                answer = left_number * right_number
            else:
                raise ValueError("Mode not known")
            samples.append((left_number, right_number, answer))
        return samples

    def generate(self, n_samples):
        samples = self.generate_calculate_random(n_samples)
        calc_sign = "+" if self.task == "sum" else "*"
        samples = [
            (
                f"calc {self.task}: "
                + str(left)
                + calc_sign
                + str(right)
                + " answer: ",
                str(answer),
            )
            for left, right, answer in samples
        ]
        return samples


class CopyTask:
    def __init__(self, min_len=10, max_len=40, max_group_length=4, reverse=False):
        self.length_low = min_len  # min length of the sequence in "words"
        self.length_high = (
            max_len + 1
        )  # max length of the sequence, +1 to include the max_len
        self.max_group_length = max_group_length  # max length of the each "word"
        self.reverse = (
            reverse  # if true, reverse the words, but not the letters in the words
        )

    def generate_random_words(self, n_samples):
        lengths = np.arange(self.length_low, self.length_high)
        group_lengths = np.arange(2, self.max_group_length + 1)
        samples = []
        for _ in range(n_samples):
            length = np.random.choice(lengths)
            group_length = np.random.choice(group_lengths)
            sample = np.random.choice(letter_chars, size=length * group_length)
            chunked = np.split(sample, length)
            string_chunks = ["".join(chunk) for chunk in chunked]
            samples.append(string_chunks)
        return samples

    def generate_copy(self, n_samples):
        samples_chuncked = self.generate_random_words(n_samples)
        samples = [" ".join(chuncks) for chuncks in samples_chuncked]
        copy_samples = [
            ("copy: " + x + " answer: ", y) for x, y in zip(samples, samples)
        ]
        return copy_samples

    def generate_reverse(self, n_samples):
        samples_chuncked = self.generate_random_words(n_samples)
        reversed_samples = [sample[::-1] for sample in samples_chuncked]
        samples = [" ".join(chuncks) for chuncks in samples_chuncked]
        reversed_samples = [" ".join(chuncks) for chuncks in reversed_samples]
        reverse_samples = [
            ("reverse: " + x + " answer: ", y)
            for x, y in zip(samples, reversed_samples)
        ]
        return reverse_samples

    def generate(self, n_samples):
        return (
            self.generate_copy(n_samples)
            if not self.reverse
            else self.generate_reverse(n_samples)
        )
