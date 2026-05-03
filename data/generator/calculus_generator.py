import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sympy as sp


class CalculusGenerator:
    def __init__(self, output_dir: str = "data/processed", target_size_gb: float = 2.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size_bytes = target_size_gb * 1024**3
        self.output_file = self.output_dir / "calculus_corpus.jsonl"
        self.lock = threading.Lock()
        self.current_size = 0

    def generate_derivative(self):
        x = sp.Symbol("x")
        # Generate random polynomial or trigonometric function
        funcs = [
            sp.sin(x) * x ** random.randint(1, 5),
            sp.cos(x) * sp.exp(x),
            x ** random.randint(2, 6) + sp.log(x),
            (x**2 + 1) / (x + 2),
        ]
        func = random.choice(funcs)
        deriv = sp.diff(func, x)

        problem = f"Compute the derivative of {func} with respect to x."
        steps = [
            f"Let f(x) = {func}",
            "We apply the appropriate differentiation rules (product rule, quotient rule, chain rule).",
            "Step 1: The derivative of the components are computed.",
            f"Step 2: Combine using rules to get {deriv}",
            f"Final Answer: {deriv}",
        ]
        solution = "\n".join(steps)
        return {"problem": problem, "solution": solution, "type": "derivative"}

    def generate_integral(self):
        x = sp.Symbol("x")
        funcs = [x ** random.randint(1, 5), sp.sin(x), sp.exp(x) * x]
        func = random.choice(funcs)
        integral = sp.integrate(func, x)

        problem = f"Compute the indefinite integral of {func} with respect to x."
        steps = [
            f"Let f(x) = {func}",
            "We want to find the antiderivative F(x) such that F'(x) = f(x).",
            f"Applying integration techniques yields {integral}.",
            f"Final Answer: {integral} + C",
        ]
        solution = "\n".join(steps)
        return {"problem": problem, "solution": solution, "type": "integral"}

    def worker_loop(self):
        while self.current_size < self.target_size_bytes:
            choice = random.choice([self.generate_derivative, self.generate_integral])
            data = choice()

            with self.lock:
                if self.current_size >= self.target_size_bytes:
                    break
                line = json.dumps(data) + "\n"
                with open(self.output_file, "a") as f:
                    f.write(line)
                self.current_size += len(line.encode("utf-8"))

    def run(self, num_threads: int = 8):
        print(f"Starting generation. Target: {self.target_size_bytes / 1024**3:.2f} GB")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.worker_loop) for _ in range(num_threads)]
            for f in futures:
                f.result()
        print("Generation complete.")


if __name__ == "__main__":
    generator = CalculusGenerator(target_size_gb=0.001)  # Small target for demo purposes
    generator.run()
