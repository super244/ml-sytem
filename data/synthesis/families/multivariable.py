from __future__ import annotations

from fractions import Fraction
import random
from typing import Any

from data.synthesis.base import DatasetSpec, frac_str, make_record, nz


def generate_multivariable_example(
    rng: random.Random,
    spec: DatasetSpec,
    index: int,
    difficulty: str,
) -> dict[str, Any]:
    mode = rng.choice(["gradient", "directional", "tangent_plane", "double_integral"])
    if mode == "gradient":
        a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), nz(rng, -4, 4)
        x0, y0 = rng.randint(-2, 3), rng.randint(-2, 3)
        question = f"For f(x,y) = {a}x^2 {b:+d}xy {c:+d}y^2, compute grad f at ({x0}, {y0})."
        fx = 2 * a * x0 + b * y0
        fy = b * x0 + 2 * c * y0
        answer = f"<{fx}, {fy}>"
        solution = (
            f"Compute partial derivatives: f_x = {2*a}x {b:+d}y and f_y = {b}x {2*c:+d}y. "
            f"Evaluate at ({x0}, {y0}) to get <{fx}, {fy}>.\nFinal Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [str(fx), str(fy)], "gradient")
    if mode == "directional":
        a, b, c = nz(rng, -3, 3), nz(rng, -3, 3), nz(rng, -3, 3)
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        u1, u2 = rng.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
        fx = 2 * a * x0 + c * y0
        fy = 2 * b * y0 + c * x0
        if abs(u1) + abs(u2) == 1:
            answer = str(fx * u1 + fy * u2)
            unit_direction = f"<{u1}, {u2}>"
        else:
            signed_numerator = fx * u1 + fy * u2
            answer = f"{signed_numerator}/sqrt(2)"
            unit_direction = f"<{u1}/sqrt(2), {u2}/sqrt(2)>"
        question = (
            f"Let f(x,y) = {a}x^2 {b:+d}y^2 {c:+d}xy. "
            f"Find the directional derivative at ({x0}, {y0}) in the direction <{u1}, {u2}>."
        )
        solution = (
            f"First compute ∇f = <{2*a}x {c:+d}y, {2*b}y {c:+d}x>. "
            f"At ({x0}, {y0}), the gradient is <{fx}, {fy}>. "
            f"The corresponding unit direction is {unit_direction}, so the directional derivative is {answer}.\n"
            f"Final Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [str(fx), str(fy), answer], "directional derivative")
    if mode == "tangent_plane":
        a, b = nz(rng, -3, 3), nz(rng, -3, 3)
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        z0 = a * x0 * x0 + b * y0 * y0 + x0 * y0
        fx = 2 * a * x0 + y0
        fy = 2 * b * y0 + x0
        answer = f"z = {z0} + {fx}(x - {x0}) + {fy}(y - {y0})"
        question = f"Find the tangent plane to z = {a}x^2 {b:+d}y^2 + xy at ({x0}, {y0})."
        solution = (
            f"For z = f(x,y), the tangent plane at ({x0}, {y0}) is "
            f"z = f(x0,y0) + f_x(x0,y0)(x-x0) + f_y(x0,y0)(y-y0). "
            f"Here f(x0,y0) = {z0}, f_x = {fx}, and f_y = {fy}. "
            f"So the plane is {answer}.\nFinal Answer: {answer}"
        )
        return make_record(spec, index, difficulty, question, solution, answer, [str(z0), str(fx), str(fy)], "tangent plane")

    a, b = nz(rng, 1, 4), nz(rng, 1, 4)
    upper_x, upper_y = rng.randint(1, 4), rng.randint(1, 4)
    answer = frac_str(Fraction(a * upper_x * upper_x * upper_y, 2) + Fraction(b * upper_x * upper_y * upper_y, 2))
    question = f"Evaluate the double integral int_0^{upper_x} int_0^{upper_y} ({a}x + {b}y) dy dx."
    solution = (
        f"Integrate with respect to y first: ∫_0^{upper_y} ({a}x + {b}y) dy = {a * upper_y}x + {frac_str(Fraction(b * upper_y * upper_y, 2))}. "
        f"Then integrate that result from x = 0 to x = {upper_x}. The value is {answer}.\n"
        f"Final Answer: {answer}"
    )
    return make_record(spec, index, difficulty, question, solution, answer, [answer], "double integral")
