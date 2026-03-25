from __future__ import annotations

import random
from typing import Any

from data.synthesis.base import DatasetSpec, derivative_terms, frac_str, make_record, nz, poly_str, poly_value


def generate_derivative_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["poly_exp", "poly_trig", "log_chain", "point_slope"])
    if mode == "poly_exp":
        a, b, c = nz(rng, 1, 5), nz(rng, -5, 5), nz(rng, 1, 4)
        n, m = rng.randint(2, 5), rng.randint(0, 3)
        poly = poly_str([(a, n), (b, m)])
        dpoly = poly_str(derivative_terms([(a, n), (b, m)]))
        derivative = f"({dpoly})e^({c}x) + {c}({poly})e^({c}x)"
        question = f"Differentiate the function f(x) = ({poly})e^({c}x)."
        solution = (
            f"Use the product rule with u(x) = {poly} and v(x) = e^({c}x). "
            f"Then u'(x) = {dpoly} and v'(x) = {c}e^({c}x). "
            f"Therefore f'(x) = u'(x)v(x) + u(x)v'(x) = ({dpoly})e^({c}x) + {c}({poly})e^({c}x).\n"
            f"Final Answer: {derivative}"
        )
        return make_record(spec, index, difficulty, question, solution, derivative, [dpoly, f"e^({c}x)"], "product rule")
    if mode == "poly_trig":
        a, n, c, d = nz(rng, 1, 5), rng.randint(2, 5), nz(rng, 1, 5), rng.randint(-4, 4)
        trig = rng.choice(["sin", "cos"])
        poly_terms = [(a, n), (nz(rng, -4, 4), 1), (rng.randint(-5, 5), 0)]
        poly = poly_str(poly_terms)
        dpoly = poly_str(derivative_terms(poly_terms))
        if trig == "sin":
            derivative = f"({dpoly})sin({c}x {d:+d}) + {c}({poly})cos({c}x {d:+d})"
        else:
            derivative = f"({dpoly})cos({c}x {d:+d}) - {c}({poly})sin({c}x {d:+d})"
        question = f"Compute d/dx of y = ({poly}){trig}({c}x {d:+d})."
        solution = (
            f"Apply the product rule. The polynomial factor differentiates to {dpoly}. "
            f"The trigonometric factor contributes a chain-rule factor of {c}. "
            f"Combining the two pieces gives the derivative below.\n"
            f"Final Answer: {derivative}"
        )
        return make_record(spec, index, difficulty, question, solution, derivative, [dpoly, str(c)], "chain rule")
    if mode == "log_chain":
        a, b = nz(rng, 1, 6), nz(rng, -6, 6)
        p_terms = [(nz(rng, -4, 4), rng.randint(2, 4)), (nz(rng, -4, 4), 1), (rng.randint(-6, 6), 0)]
        poly = poly_str(p_terms)
        dpoly = poly_str(derivative_terms(p_terms))
        derivative = f"{dpoly} + {a}/({a}x {b:+d})"
        question = f"Differentiate g(x) = ln({a}x {b:+d}) + {poly}."
        solution = (
            f"The derivative of ln({a}x {b:+d}) is {a}/({a}x {b:+d}) by the chain rule. "
            f"The derivative of the polynomial part is {dpoly}. "
            f"Add the two contributions.\nFinal Answer: {derivative}"
        )
        return make_record(spec, index, difficulty, question, solution, derivative, [dpoly, f"{a}/({a}x {b:+d})"], "logarithmic differentiation")

    point = rng.randint(-2, 3)
    terms = [(nz(rng, -4, 4), rng.randint(2, 5)), (nz(rng, -4, 4), 1), (rng.randint(-5, 5), 0)]
    poly = poly_str(terms)
    dterms = derivative_terms(terms)
    slope = poly_value(dterms, point)
    derivative = frac_str(slope)
    question = f"For h(x) = {poly}, compute h'({point})."
    solution = (
        f"Differentiate h(x) term by term to get h'(x) = {poly_str(dterms)}. "
        f"Substitute x = {point}: h'({point}) = {frac_str(slope)}.\n"
        f"Final Answer: {derivative}"
    )
    return make_record(spec, index, difficulty, question, solution, derivative, [poly_str(dterms), derivative], "pointwise derivative evaluation")
