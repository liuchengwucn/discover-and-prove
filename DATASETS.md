# Datasets

This repository ships four evaluation datasets used by the Discover and Prove
(DAP) framework. Two of them (**MiniF2F-Hard**, **FIMO-Hard**) are our
expert-reannotated Hard Mode variants; the other two (**PutnamBench**,
**CombiBench**) are used as-is from upstream, which already provide Hard Mode
("no answer" / "without solution") formalizations.

Each dataset is a derived work of an upstream benchmark and remains under its
**upstream license**. Please cite and comply with the original sources.

| Dataset (here)            | Files                                   | Upstream source | Upstream license |
|---------------------------|-----------------------------------------|-----------------|------------------|
| PutnamBench               | `datasets/putnambench/`                 | [trishullab/PutnamBench](https://github.com/trishullab/PutnamBench) | Apache-2.0 |
| CombiBench                | `datasets/combibench/`                  | [MoonshotAI/CombiBench](https://github.com/MoonshotAI/CombiBench) | MIT |
| MiniF2F-Hard (ours)       | `datasets/minif2f_hard.json`            | [openai/miniF2F](https://github.com/openai/miniF2F) (Lean 4 community ports) | MIT |
| FIMO-Hard (ours)          | `datasets/fimo_hard.json`               | [liuchengwucn/FIMO](https://github.com/liuchengwucn/FIMO) | MIT |

## Schema

**`minif2f_hard.json`, `fimo_hard.json`** — list of objects:

- `id` — problem identifier
- `math_problem` — natural-language problem statement
- `solution` — ground-truth final answer (`null` for proof-style problems)
- `formal_statement` — Lean 4 Hard Mode statement (two `sorry`s for solution-style)
- `formal_statement_easy` — Lean 4 Easy Mode statement (FIMO only)
- `is_hard` — whether the problem is a Hard Mode (solution-style) problem

**`putnambench/`** — `formal/*.lean` (Hard Mode statements; answer-bearing
comment lines starting with `--` are stripped at load time) and `informal.json`
(`problem_name`, `informal_statement`, `informal_solution`).

**`combibench/`** — `formal/*.lean` and `informal.csv` (`theorem_name`,
`natural_language`, `answer`, ...).

## Reannotation & quality fixes

- **MiniF2F-Hard** contains **197** Hard Mode problems. Relative to an earlier
  194-problem version, three IMO problems were promoted to Hard Mode:
  `imo_1960_p2` and `imo_1981_p6` (previously mislabeled solution-style problems
  whose answers — a solution set and a closed-form value — must be discovered),
  and `imo_1983_p6` (whose "determine when equality occurs" answer, `a = b = c`,
  is treated as the quantity to discover). The paper's main table and Table 3
  use 197; Table 2 and the self-verification ablation report the older 194.

- **Answer-leak fixes (FIMO-Hard):** three problems had the final answer
  embedded in the natural-language statement; the leaked text was removed so the
  answer must be discovered:
  `fimo_2010_number_theory_p1_2`, `fimo_2011_algebra_p2`,
  `fimo_2012_number_theory_p1`. As a result these statements differ slightly
  from the exact inputs used to produce the paper's numbers.
