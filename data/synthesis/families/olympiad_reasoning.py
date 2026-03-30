from __future__ import annotations

import random
from typing import Any

from data.synthesis.base import DatasetSpec, make_record, nz


def generate_olympiad_reasoning_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["divisibility", "functional_pattern", "counting", "inequality"])
    if mode == "divisibility":
        n = rng.randint(2, 8)
        question = f"Show that n^3 - n is divisible by 6 for n = {n}, then state the general pattern."
        answer = "n^3 - n is divisible by 6 for every integer n"
        solution = (
            f"Factor n^3 - n as n(n-1)(n+1), the product of three consecutive integers. "
            f"Among three consecutive integers, one is divisible by 3 and at least one is even, "
            f"so the product is divisible by 6. For n = {n}, this confirms the same pattern.\n"
            f"Final Answer: {answer}"
        )
        return make_record(
            spec,
            index,
            difficulty,
            question,
            solution,
            answer,
            ["n(n-1)(n+1)", "divisible by 6"],
            "proof by factorization",
            topic="number theory",
            reasoning_style="proof",
            tags=["olympiad", "proof_style"],
        )
    if mode == "functional_pattern":
        a = nz(rng, 1, 6)
        question = f"If f(x) = x^2 + {a}x + 1, compute f(x+1) - f(x) and identify the linear pattern."
        answer = f"2x + {a + 1}"
        solution = (
            f"Expand f(x+1) = (x+1)^2 + {a}(x+1) + 1 = x^2 + ({a + 2})x + {a + 2}. "
            f"Subtract f(x) = x^2 + {a}x + 1 to get 2x + {a + 1}. "
            f"This shows the first finite difference is linear.\n"
            f"Final Answer: {answer}"
        )
        return make_record(
            spec,
            index,
            difficulty,
            question,
            solution,
            answer,
            [answer, "linear"],
            "algebraic pattern recognition",
            topic="algebra",
            reasoning_style="exam",
            tags=["olympiad", "functional"],
        )
    if mode == "counting":
        n = rng.randint(4, 9)
        question = f"How many diagonals does a convex {n}-gon have? Justify the counting formula."
        answer = str(n * (n - 3) // 2)
        solution = (
            f"Each vertex connects to {n - 3} diagonals because it cannot connect to itself or its two adjacent vertices. "
            f"This counts every diagonal twice, once from each endpoint, so the number is n(n-3)/2 = {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(
            spec,
            index,
            difficulty,
            question,
            solution,
            answer,
            ["counted twice", answer],
            "double counting",
            topic="combinatorics",
            reasoning_style="proof",
            tags=["olympiad", "counting"],
        )

    a = rng.randint(1, 4)
    b = rng.randint(1, 5)
    question = f"Prove that x^2 + {a}x + {b} >= {b - a * a / 4:.2f} for all real x, and identify when equality holds."
    answer = f"x = {-a}/2"
    solution = (
        f"Complete the square: x^2 + {a}x + {b} = (x + {a}/2)^2 + {b} - {a * a}/4. "
        f"Since the square term is always nonnegative, the expression is at least {b - a * a / 4:.2f}. "
        f"Equality holds when x + {a}/2 = 0, so x = {-a}/2.\n"
        f"Final Answer: {answer}"
    )
    return make_record(
        spec,
        index,
        difficulty,
        question,
        solution,
        answer,
        ["complete the square", answer],
        "inequality by completing the square",
        topic="algebra",
        reasoning_style="proof",
        tags=["olympiad", "inequality"],
    )
