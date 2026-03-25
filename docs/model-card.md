# Model Card

## Model Family

Atlas Math Lab is a specialist adaptation stack for competitive mathematics and advanced calculus reasoning built on top of math-capable base LLMs.

## Intended Use

- advanced calculus tutoring and experimentation
- contest-style reasoning demos
- benchmark and ablation studies for math-specialist fine-tuning

## Training Data

- local synthetic calculus and olympiad corpora
- normalized public math datasets
- optional failure replay from prior evaluation runs

## Limitations

- verification is heuristic and lightweight, not a proof assistant
- symbolic correctness is incomplete
- the system is not designed for high-stakes educational or grading decisions
- model behavior still depends heavily on prompt choice and base-model quality
