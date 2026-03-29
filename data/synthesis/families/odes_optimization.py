from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

from data.synthesis.base import DatasetSpec, frac_str, make_record, nz


def generate_odes_optimization_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["separable_ode", "exp_growth", "critical_point", "interval_extrema"])
    if mode == "separable_ode":
        k = nz(rng, -4, 4)
        y0 = nz(rng, 1, 6)
        answer = f"y = {y0}e^({k}x)"
        question = f"Solve the initial value problem y' = {k}y, y(0) = {y0}."
        solution = (
            f"Separate variables: dy/y = {k} dx. Integrating gives ln|y| = {k}x + C, so y = Ce^({k}x). "
            f"Use y(0) = {y0} to get C = {y0}. Therefore y = {y0}e^({k}x).\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [str(y0), str(k)], "separable ODE")
    if mode == "exp_growth":
        rate = Fraction(rng.randint(1, 5), rng.randint(2, 6))
        initial = rng.randint(10, 60)
        time_value = rng.randint(1, 5)
        answer = f"{initial}e^({frac_str(rate)}*{time_value})"
        question = f"A quantity follows P'(t) = {frac_str(rate)}P(t) with P(0) = {initial}. Find P({time_value})."
        solution = (
            f"The differential equation has solution P(t) = {initial}e^({frac_str(rate)}t). "
            f"Substitute t = {time_value} to obtain P({time_value}) = {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [str(initial), frac_str(rate)], "exponential growth")
    if mode == "critical_point":
        a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), rng.randint(-5, 5)
        if a == 0:
            a = 2
        vertex = Fraction(-b, 2 * a)
        question = f"Find the critical point of f(x) = {a}x^2 {b:+d}x {c:+d}."
        answer = frac_str(vertex)
        solution = (
            f"Differentiate: f'(x) = {2*a}x {b:+d}. "
            f"Set f'(x) = 0 and solve for x, giving x = {-b}/({2*a}) = {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [f"{2*a}x {b:+d}", answer], "critical point")

    a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), rng.randint(-6, 6)
    left, right = sorted((rng.randint(-3, 0), rng.randint(1, 4)))
    if a == 0:
        a = 1
    vertex = Fraction(-b, 2 * a)
    candidates = [Fraction(left, 1), Fraction(right, 1)]
    if left <= vertex <= right:
        candidates.append(vertex)
    values = {candidate: a * candidate * candidate + b * candidate + c for candidate in candidates}
    best_x = min(values, key=values.get) if a > 0 else max(values, key=values.get)
    answer = f"x = {frac_str(best_x)}"
    question = (
        f"On the interval [{left}, {right}], where does f(x) = {a}x^2 {b:+d}x {c:+d} "
        f"attain its {'minimum' if a > 0 else 'maximum'}?"
    )
    solution = (
        f"Check the endpoints and any critical point in the interval. "
        f"Since f'(x) = {2*a}x {b:+d}, the interior critical point is x = {frac_str(vertex)}. "
        f"Comparing endpoint values with the critical value shows the extremum occurs at {answer}.\n"
        f"Final Answer: {answer}"
    )
    return make_record(spec, index, difficulty, question, solution, answer, [frac_str(vertex), answer], "interval optimization")
