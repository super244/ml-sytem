from __future__ import annotations

import random
from typing import Any

from data.synthesis.base import (
    DatasetSpec,
    evaluate_antiderivative,
    frac_str,
    fractional_poly_str,
    integral_terms,
    make_record,
    nz,
    poly_str,
)


def generate_integral_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["poly_definite", "poly_indefinite", "exp_sub", "trig"])
    if mode == "poly_definite":
        terms = [(nz(rng, -5, 5), rng.randint(1, 4)), (nz(rng, -4, 4), 0)]
        upper = rng.randint(1, 4)
        poly = poly_str(terms)
        antiderivative = integral_terms(terms)
        result = evaluate_antiderivative(antiderivative, upper) - evaluate_antiderivative(antiderivative, 0)
        anti_text = fractional_poly_str(antiderivative)
        final_answer = frac_str(result)
        question = f"Evaluate the definite integral int_0^{upper} ({poly}) dx."
        solution = (
            f"An antiderivative of {poly} is {anti_text}. "
            f"Using the Fundamental Theorem of Calculus, evaluate at the bounds 0 and {upper}. "
            f"The result is {final_answer}.\nFinal Answer: {final_answer}"
        )
        return make_record(spec, index, difficulty, question, solution, final_answer, [anti_text, final_answer], "definite integration")
    if mode == "poly_indefinite":
        terms = [(nz(rng, -5, 5), rng.randint(2, 5)), (nz(rng, -5, 5), rng.randint(0, 1))]
        poly = poly_str(terms)
        antiderivative = fractional_poly_str(integral_terms(terms))
        final_answer = f"{antiderivative} + C"
        question = f"Find an antiderivative of f(x) = {poly}."
        solution = (
            f"Integrate term by term. Each x^n term becomes x^(n+1)/(n+1). "
            f"That gives the antiderivative {antiderivative} + C.\n"
            f"Final Answer: {final_answer}"
        )
        return make_record(spec, index, difficulty, question, solution, final_answer, [antiderivative], "termwise antiderivative")
    if mode == "exp_sub":
        a, b = nz(rng, 1, 5), nz(rng, -5, 5)
        question = f"Evaluate int ({a})e^({a}x {b:+d}) dx."
        final_answer = f"e^({a}x {b:+d}) + C"
        solution = (
            f"Let u = {a}x {b:+d}. Then du = {a} dx, so the integral becomes int e^u du. "
            f"This integrates to e^u + C = e^({a}x {b:+d}) + C.\n"
            f"Final Answer: {final_answer}"
        )
        return make_record(spec, index, difficulty, question, solution, final_answer, [f"u = {a}x {b:+d}", "e^u"], "substitution")

    a = nz(rng, 1, 6)
    trig = rng.choice(["sin", "cos"])
    if trig == "sin":
        final_answer = f"-cos({a}x)/{a} + C"
    else:
        final_answer = f"sin({a}x)/{a} + C"
    question = f"Compute int {trig}({a}x) dx."
    solution = (
        f"Use the standard antiderivative for {trig}(ax). "
        f"A factor of 1/{a} appears because the inside derivative is {a}. "
        f"The integral equals {final_answer}.\nFinal Answer: {final_answer}"
    )
    return make_record(spec, index, difficulty, question, solution, final_answer, [str(a), final_answer], "trigonometric integral")
