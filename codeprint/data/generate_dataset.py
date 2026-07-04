"""Synthetic demo corpus: 'human student' vs 'AI assistant' solutions.

This exists so the whole pipeline runs offline with zero data collection.
The style contrasts are modeled on published observations about LLM code
(uniform formatting, docstring/type-hint saturation, consistent naming) vs
classroom code (style drift, debug residue, commented-out code, magic
numbers). For production use, replace with a real corpus: the course's own
historical submissions (pre-2022 semesters are guaranteed human) plus
LLM-generated solutions to the same assignment prompts.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

# ---------------------------------------------------------------- helpers

HUMAN_COMMENTS = [
    "# my solution", "# hw3", "# not sure if this works", "# test",
    "# TODO clean this up", "# fixme later", "# this took forever",
    "# copied from my notes", "# idk why this works", "# debug",
]

MESSY_NAMES = ["x", "n", "num", "temp", "tmp", "val", "res", "ans", "lst",
               "arr", "cnt", "i", "j", "thing", "stuff", "data1", "myList",
               "theResult", "finalAns"]

CLEAN_POOLS = {
    "count": ["count", "total_count", "num_items"],
    "result": ["result", "output", "results"],
    "values": ["values", "numbers", "items"],
    "text": ["text", "input_text", "content"],
}


def _eq(rng: random.Random) -> str:
    """Human spacing around '=' — inconsistent within a file."""
    return rng.choice(["=", "= ", " =", " = ", " = ", " = "])


def _q(rng: random.Random) -> str:
    return rng.choice(["'", '"'])


def messify(rng: random.Random, lines: list[str]) -> str:
    """Inject human residue: stray comments, commented-out code, blank lines,
    trailing whitespace. Comments/blanks are legal at any indentation, so
    insertion anywhere keeps the file parseable."""
    out: list[str] = []
    for ln in lines:
        if rng.random() < 0.10:
            out.append(rng.choice(HUMAN_COMMENTS))
        if rng.random() < 0.06:
            out.append("")
        if rng.random() < 0.08 and ln.strip():
            ln = ln + " " * rng.randint(1, 3)
        out.append(ln)
    if rng.random() < 0.5:
        out.append(rng.choice(["#print(res)", "# print(ans)", "#test code below",
                               "#for i in range(5): print(i)"]))
    return "\n".join(out) + "\n"


def clean(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- tasks
# Each task has a human(rng) and an ai(rng) generator returning source text.

def human_fib(rng: random.Random) -> str:
    n = rng.choice(["n", "num", "x"])
    a, b = rng.choice([("a", "b"), ("x", "y"), ("t1", "t2"), ("first", "second")])
    res = rng.choice(["res", "ans", "fibs", "l"])
    if rng.random() < 0.5:
        body = [
            f"def fib({n}):",
            f"    {res}{_eq(rng)}[]",
            f"    {a}{_eq(rng)}0",
            f"    {b}{_eq(rng)}1",
            f"    for i in range({n}):",
            f"        {res}.append({a})",
            f"        {a}, {b}{_eq(rng)}{b}, {a}+{b}",
            f"    return {res}",
            "",
            f"print(fib({rng.randint(8, 15)}))",
        ]
    else:
        body = [
            f"def fib({n}):",
            f"    if {n}<=1:",
            f"        return {n}",
            f"    return fib({n}-1)+fib({n}-2)",
            "",
            f"for i in range({rng.randint(8, 12)}):",
            "    print(fib(i))",
        ]
    return messify(rng, body)


def ai_fib(rng: random.Random) -> str:
    fn = rng.choice(["fibonacci", "generate_fibonacci", "fibonacci_sequence"])
    doc = rng.choice([
        "Generate the first n Fibonacci numbers.",
        "Return a list containing the first n Fibonacci numbers.",
        "Compute the Fibonacci sequence up to n terms.",
    ])
    var = rng.choice(["sequence", "fib_numbers", "result"])
    n_demo = rng.randint(8, 15)
    return clean([
        f"def {fn}(n: int) -> list[int]:",
        f'    """{doc}',
        "",
        "    Args:",
        "        n: The number of Fibonacci terms to generate.",
        "",
        "    Returns:",
        "        A list of the first n Fibonacci numbers.",
        '    """',
        f"    {var}: list[int] = []",
        "    current, nxt = 0, 1",
        "    for _ in range(n):",
        f"        {var}.append(current)",
        "        current, nxt = nxt, current + nxt",
        f"    return {var}",
        "",
        "",
        'if __name__ == "__main__":',
        f"    print({fn}({n_demo}))",
    ])


def human_primes(rng: random.Random) -> str:
    n = rng.choice(["n", "num", "limit", "x"])
    flag = rng.choice(["flag", "isp", "ok", "prime"])
    body = [
        f"{n}{_eq(rng)}{rng.randint(30, 80)}",
        f"for i in range(2, {n}):",
        f"    {flag}{_eq(rng)}True",
        "    for j in range(2, i):",
        "        if i%j==0:",
        f"            {flag}{_eq(rng)}False",
        "            break" if rng.random() < 0.5 else "    # break here?",
        f"    if {flag}:",
        "        print(i)",
    ]
    body = [ln for ln in body if ln.strip() != "# break here?"] \
        if body[6].strip().startswith("#") and rng.random() < 0.5 else body
    return messify(rng, body)


def ai_primes(rng: random.Random) -> str:
    fn = rng.choice(["is_prime", "check_prime"])
    doc = rng.choice([
        "Check whether a number is prime.",
        "Determine if the given integer is a prime number.",
    ])
    limit = rng.randint(30, 80)
    return clean([
        "import math",
        "",
        "",
        f"def {fn}(number: int) -> bool:",
        f'    """{doc}"""',
        "    if number < 2:",
        "        return False",
        "    for divisor in range(2, int(math.isqrt(number)) + 1):",
        "        if number % divisor == 0:",
        "            return False",
        "    return True",
        "",
        "",
        "def get_primes(limit: int) -> list[int]:",
        '    """Return all prime numbers up to the given limit."""',
        f"    return [num for num in range(2, limit) if {fn}(num)]",
        "",
        "",
        'if __name__ == "__main__":',
        f"    print(get_primes({limit}))",
    ])


def human_wordfreq(rng: random.Random) -> str:
    d = rng.choice(["d", "counts", "freq", "wc"])
    w = rng.choice(["w", "word", "x"])
    s = rng.choice(["s", "text", "sentence"])
    q = _q(rng)
    body = [
        f"{s}{_eq(rng)}{q}the quick brown fox jumps over the lazy dog the fox{q}",
        f"{d}{_eq(rng)}{{}}",
        f"for {w} in {s}.split():",
        f"    if {w} in {d}:",
        f"        {d}[{w}]{_eq(rng)}{d}[{w}]+1",
        "    else:",
        f"        {d}[{w}]{_eq(rng)}1",
        f"print({d})",
    ]
    return messify(rng, body)


def ai_wordfreq(rng: random.Random) -> str:
    fn = rng.choice(["count_words", "word_frequency", "get_word_counts"])
    doc = rng.choice([
        "Count the frequency of each word in the given text.",
        "Return a dictionary mapping each word to its frequency.",
    ])
    return clean([
        "from collections import Counter",
        "",
        "",
        f"def {fn}(text: str) -> dict[str, int]:",
        f'    """{doc}',
        "",
        "    Args:",
        "        text: The input text to analyze.",
        "",
        "    Returns:",
        "        A dictionary mapping words to their occurrence counts.",
        '    """',
        "    words = text.lower().split()",
        "    return dict(Counter(words))",
        "",
        "",
        'if __name__ == "__main__":',
        '    sample_text = "the quick brown fox jumps over the lazy dog the fox"',
        f"    frequencies = {fn}(sample_text)",
        "    for word, count in sorted(frequencies.items()):",
        '        print(f"{word}: {count}")',
    ])


def human_palindrome(rng: random.Random) -> str:
    s = rng.choice(["s", "word", "txt", "string1"])
    q = _q(rng)
    if rng.random() < 0.5:
        body = [
            f"{s}{_eq(rng)}input({q}enter a word: {q})" if rng.random() < 0.4
            else f"{s}{_eq(rng)}{q}racecar{q}",
            f"if {s}=={s}[::-1]:",
            f"    print({q}palindrome{q})",
            "else:",
            f"    print({q}not a palindrome{q})",
        ]
    else:
        r = rng.choice(["rev", "r", "backwards"])
        body = [
            f"{s}{_eq(rng)}{q}racecar{q}",
            f"{r}{_eq(rng)}{q}{q}",
            f"for c in {s}:",
            f"    {r}{_eq(rng)}c+{r}",
            f"print({s}=={r})",
        ]
    return messify(rng, body)


def ai_palindrome(rng: random.Random) -> str:
    fn = rng.choice(["is_palindrome", "check_palindrome"])
    doc = rng.choice([
        "Check whether the given string is a palindrome.",
        "Determine if a string reads the same forwards and backwards.",
    ])
    word = rng.choice(["racecar", "level", "hello"])
    return clean([
        f"def {fn}(text: str) -> bool:",
        f'    """{doc}',
        "",
        "    Comparison is case-insensitive and ignores spaces.",
        '    """',
        '    normalized = text.lower().replace(" ", "")',
        "    return normalized == normalized[::-1]",
        "",
        "",
        'if __name__ == "__main__":',
        f'    test_word = "{word}"',
        f'    if {fn}(test_word):',
        '        print(f"{test_word!r} is a palindrome")',
        "    else:",
        '        print(f"{test_word!r} is not a palindrome")',
    ])


def human_grades(rng: random.Random) -> str:
    g = rng.choice(["grades", "marks", "scores", "l"])
    t = rng.choice(["total", "sum1", "s", "tot"])
    nums = ", ".join(str(rng.randint(40, 100)) for _ in range(rng.randint(5, 8)))
    q = _q(rng)
    body = [
        f"{g}{_eq(rng)}[{nums}]",
        f"{t}{_eq(rng)}0",
        f"for x in {g}:",
        f"    {t}{_eq(rng)}{t}+x",
        f"avg{_eq(rng)}{t}/len({g})",
        f"print({q}average is{q}, avg)",
        "if avg>=60:" if rng.random() < 0.6 else "if avg >= 50:",
        f"    print({q}pass{q})",
        "else:",
        f"    print({q}fail{q})",
    ]
    return messify(rng, body)


def ai_grades(rng: random.Random) -> str:
    fn = rng.choice(["calculate_average", "compute_average", "get_average_grade"])
    threshold = rng.choice([50, 60])
    nums = ", ".join(str(rng.randint(40, 100)) for _ in range(rng.randint(5, 8)))
    return clean([
        f"def {fn}(grades: list[float]) -> float:",
        '    """Calculate the average of a list of grades.',
        "",
        "    Args:",
        "        grades: A list of numeric grade values.",
        "",
        "    Returns:",
        "        The arithmetic mean of the grades.",
        "",
        "    Raises:",
        "        ValueError: If the grades list is empty.",
        '    """',
        "    if not grades:",
        '        raise ValueError("Cannot compute the average of an empty list")',
        "    return sum(grades) / len(grades)",
        "",
        "",
        "def determine_status(average: float, passing_threshold: float = "
        f"{threshold}.0) -> str:",
        '    """Return \'pass\' or \'fail\' based on the average grade."""',
        '    return "pass" if average >= passing_threshold else "fail"',
        "",
        "",
        'if __name__ == "__main__":',
        f"    student_grades = [{nums}]",
        f"    average_grade = {fn}(student_grades)",
        '    print(f"Average grade: {average_grade:.2f}")',
        '    print(f"Status: {determine_status(average_grade)}")',
    ])


def human_bank(rng: random.Random) -> str:
    cls = rng.choice(["Bank", "Account", "bankAccount", "MyBank"])
    bal = rng.choice(["bal", "balance", "money", "b"])
    q = _q(rng)
    body = [
        f"class {cls}:",
        f"    def __init__(self):",
        f"        self.{bal}{_eq(rng)}0",
        "    def deposit(self, amt):",
        f"        self.{bal}{_eq(rng)}self.{bal}+amt",
        "    def withdraw(self, amt):",
        f"        if amt>self.{bal}:",
        f"            print({q}not enough money{q})",
        "        else:",
        f"            self.{bal}{_eq(rng)}self.{bal}-amt",
        "",
        f"acc{_eq(rng)}{cls}()",
        f"acc.deposit({rng.randint(50, 500)})",
        f"acc.withdraw({rng.randint(10, 100)})",
        f"print(acc.{bal})",
    ]
    return messify(rng, body)


def ai_bank(rng: random.Random) -> str:
    deposit_amt = rng.randint(50, 500)
    withdraw_amt = rng.randint(10, 100)
    return clean([
        "class BankAccount:",
        '    """A simple bank account supporting deposits and withdrawals."""',
        "",
        "    def __init__(self, initial_balance: float = 0.0) -> None:",
        '        """Initialize the account with an optional starting balance."""',
        "        self._balance = initial_balance",
        "",
        "    @property",
        "    def balance(self) -> float:",
        '        """Return the current account balance."""',
        "        return self._balance",
        "",
        "    def deposit(self, amount: float) -> None:",
        '        """Deposit a positive amount into the account.',
        "",
        "        Raises:",
        "            ValueError: If the amount is not positive.",
        '        """',
        "        if amount <= 0:",
        '            raise ValueError("Deposit amount must be positive")',
        "        self._balance += amount",
        "",
        "    def withdraw(self, amount: float) -> None:",
        '        """Withdraw an amount from the account.',
        "",
        "        Raises:",
        "            ValueError: If funds are insufficient or the amount is invalid.",
        '        """',
        "        if amount <= 0:",
        '            raise ValueError("Withdrawal amount must be positive")',
        "        if amount > self._balance:",
        '            raise ValueError("Insufficient funds")',
        "        self._balance -= amount",
        "",
        "",
        'if __name__ == "__main__":',
        "    account = BankAccount()",
        f"    account.deposit({deposit_amt})",
        f"    account.withdraw({withdraw_amt})",
        '    print(f"Final balance: {account.balance:.2f}")',
    ])


def human_temps(rng: random.Random) -> str:
    c = rng.choice(["c", "cel", "temp", "t"])
    f = rng.choice(["f", "fahr", "temp2", "result"])
    q = _q(rng)
    body = [
        f"for {c} in range(0, {rng.choice([50, 100, 101])}, {rng.choice([5, 10])}):",
        f"    {f}{_eq(rng)}{c}*9/5+32",
        f"    print({c}, {q}->{q}, {f})",
    ]
    return messify(rng, body)


def ai_temps(rng: random.Random) -> str:
    fn = rng.choice(["celsius_to_fahrenheit", "convert_celsius_to_fahrenheit"])
    step = rng.choice([5, 10])
    return clean([
        f"def {fn}(celsius: float) -> float:",
        '    """Convert a temperature from Celsius to Fahrenheit.',
        "",
        "    Args:",
        "        celsius: Temperature in degrees Celsius.",
        "",
        "    Returns:",
        "        The equivalent temperature in degrees Fahrenheit.",
        '    """',
        "    return celsius * 9 / 5 + 32",
        "",
        "",
        "def print_conversion_table(start: int = 0, stop: int = 100, "
        f"step: int = {step}) -> None:",
        '    """Print a Celsius-to-Fahrenheit conversion table."""',
        "    for celsius in range(start, stop + 1, step):",
        f"        fahrenheit = {fn}(celsius)",
        '        print(f"{celsius:>5}°C = {fahrenheit:>7.1f}°F")',
        "",
        "",
        'if __name__ == "__main__":',
        "    print_conversion_table()",
    ])


TASKS = {
    "fibonacci": (human_fib, ai_fib),
    "primes": (human_primes, ai_primes),
    "word_frequency": (human_wordfreq, ai_wordfreq),
    "palindrome": (human_palindrome, ai_palindrome),
    "grades": (human_grades, ai_grades),
    "bank_account": (human_bank, ai_bank),
    "temperature": (human_temps, ai_temps),
}


# ---------------------------------------------------------------- assembly

def _make_sample(rng: random.Random, task: str, label: int) -> str:
    gen = TASKS[task][label]
    for _ in range(10):
        code = gen(rng)
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue
    raise RuntimeError(f"could not generate valid sample for {task} label={label}")


def generate(out_path: str | Path, n_students: int = 80, seed: int = 7) -> int:
    """~70% honest students (human code), ~30% AI-submitting students."""
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_names = list(TASKS)
    samples = []
    n_honest = int(n_students * 0.7)
    for s in range(n_students):
        student = f"s{s:03d}"
        label = 0 if s < n_honest else 1
        for task in rng.sample(task_names, k=rng.randint(4, len(task_names))):
            samples.append({
                "student_id": student,
                "assignment": task,
                "label": label,
                "code": _make_sample(rng, task, label),
            })

    rng.shuffle(samples)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in samples:
            fh.write(json.dumps(rec) + "\n")
    return len(samples)


if __name__ == "__main__":
    print(generate("data/dataset.jsonl"))
