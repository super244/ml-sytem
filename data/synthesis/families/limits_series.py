from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

from data.synthesis.base import DatasetSpec, frac_str, make_record, nz


def generate_limits_series_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["limit_sin", "limit_exp", "limit_ratio", "p_series", "geometric_series"])
    if mode == "limit_sin":
        a, b = nz(rng, 1, 6), nz(rng, 1, 6)
        answer = frac_str(Fraction(a, b))
        question = f"Evaluate lim_(x->0) sin({a}x)/({b}x)."
        solution = (
            f"Use the standard limit sin(t)/t -> 1 as t -> 0 with t = {a}x. "
            f"Then sin({a}x)/({b}x) = ({a}/{b}) * sin({a}x)/({a}x), so the limit is {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [f"{a}/{b}", answer], "standard limit")
    if mode == "limit_exp":
        a = nz(rng, 1, 6)
        question = f"Evaluate lim_(x->0) (e^({a}x) - 1)/x."
        answer = str(a)
        solution = (
            f"Differentiate the numerator and denominator or recall that (e^(ax) - 1)/x tends to a as x -> 0. "
            f"Here a = {a}, so the limit is {a}.\nFinal Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [answer], "l'Hospital style limit")
    if mode == "limit_ratio":
        a, b, c = nz(rng, -5, 5), nz(rng, -5, 5), rng.randint(-5, 5)
        d, e, f = nz(rng, -5, 5), nz(rng, -5, 5), rng.randint(-5, 5)
        answer = frac_str(Fraction(a, d))
        question = f"Compute lim_(x->infinity) ({a}x^2 {b:+d}x {c:+d})/({d}x^2 {e:+d}x {f:+d})."
        solution = (
            f"For rational functions of the same degree, the limit at infinity is the ratio of leading coefficients. "
            f"That ratio is {a}/{d} = {answer}.\nFinal Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [f"{a}/{d}", answer], "asymptotic ratio")
    if mode == "p_series":
        p = rng.choice([Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(5, 4), Fraction(2, 1), Fraction(3, 1)])
        answer = "converges" if p > 1 else "diverges"
        question = f"Does the series sum_(n=1)^infinity 1/n^({frac_str(p)}) converge or diverge?"
        solution = (
            f"This is a p-series with p = {frac_str(p)}. "
            f"A p-series converges exactly when p > 1. Therefore the series {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [f"p = {frac_str(p)}", answer], "convergence test")

    ratio = Fraction(rng.choice([1, 2, 3, 4]), rng.choice([5, 6, 7, 8]))
    answer = frac_str(Fraction(1, 1) / (1 - ratio))
    question = f"Find the sum of the infinite geometric series 1 + {frac_str(ratio)} + {frac_str(ratio)}^2 + ..."
    solution = (
        f"The common ratio is r = {frac_str(ratio)}, which satisfies |r| < 1. "
        f"So the sum is 1/(1-r) = {answer}.\nFinal Answer: {answer}"
    )
    return make_record(spec, index, difficulty, question, solution, answer, [f"r = {frac_str(ratio)}", answer], "geometric series")
