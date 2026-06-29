# Discover and Prove (ACL 2026 Main)

**TL;DR**: Hard Mode automated theorem proving in Lean 4 — the system must first *discover* the final answer in natural language, then *prove* it formally.

The official implementation of our paper **Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4** and its associated Hard Mode benchmarks **MiniF2F-Hard** and **FIMO-Hard**.

<p align="center">
  📃 <a href="https://aclanthology.org/2026.acl-long.3/" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/liuchengwucn/discover-and-prove" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/datasets/liuchengwu/discover-and-prove" target="_blank">[Dataset]</a>
</p>

In **Hard Mode**, the final answer is **not** embedded in the formal statement:
the system must first *discover* the answer in natural language and only then
*prove* it formally — mirroring what a human competitor actually faces. DAP does
this with two decoupled modules:

- **Discovery Module** — a reasoning LLM (gpt-oss-120b) solves the problem in
  natural language with iterative self-verification and self-correction, then
  rewrites the Hard Mode Lean 4 statement (two `sorry`s) into an Easy Mode one
  (one `sorry`).
- **Proving Module** — a state-of-the-art prover (Goedel-Prover-V2-32B) closes
  the remaining goal; proofs are verified through a Kimina server / Lean 4 REPL.

## Repository layout

```
run.py                 CLI entry point (main / answer-acc / rewrite-ablation)
discover_and_prove.py  Core pipeline: discover -> rewrite -> prove
agent.py               Discovery Module (generation / self-verification / self-correction)
goedel_wrapper.py      Proving Module: prover sampling + Kimina (Lean 4) checking
judge.py               Table 3: answer-accuracy LLM-as-judge
models.py              Reasoning LLM + judge clients (OpenAI-compatible)
data_loader.py         Dataset and ground-truth-answer loaders
prompts/               Prompts (generation, verification, correction, rewrite, judge)
datasets/              PutnamBench, CombiBench, MiniF2F-Hard, FIMO-Hard (see DATASETS.md)
```

## Setup

Requires Python ≥ 3.12. With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Copy `.env.example` to `.env` and configure the three services DAP talks to:

```bash
cp .env.example .env
```

| Service | Env vars | What it is |
|---|---|---|
| Reasoning LLM + judge | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI-compatible endpoint serving the discovery model (`openai/gpt-oss-120b`) and the judge (`google/gemini-2.5-flash`). Model names are OpenRouter-style. |
| Goedel-Prover-V2 | `GOEDEL_BASE_URL`, `GOEDEL_API_KEY`, `GOEDEL_MODEL_ID` | Self-hosted prover (e.g. vLLM serving `Goedel-LM/Goedel-Prover-V2-32B`). |
| Kimina server | `KIMINA_API_URL` | Mediates the Lean 4 REPL for proof checking (we use Lean 4.15.0). |

You must provide these yourself; see the upstream projects for
[Goedel-Prover-V2](https://github.com/Goedel-LM/Goedel-Prover) and
[Kimina](https://github.com/project-numina/kimina-lean-server).

## Reproducing the paper

All runs are Pass@32 (developer-recommended prover sampling). LLM sampling is
stochastic, so exact counts vary slightly between runs. Use `--ratio` for quick
smoke tests on a fraction of a dataset. `--dataset` is one of
`putnam | combi | minif2f | fimo`.

**Table 1 — main results (DAP w/ and w/o Agent):**

```bash
python run.py main --dataset putnam              # DAP w/ Agent
python run.py main --dataset putnam --no-agent   # DAP w/o Agent
```

Each run writes `eval_dap_on_<dataset>_intermediate_<ratio>[...].json` (the
`minif2f` and `fimo` datasets use the tokens `minif2f_annotated` / `fimo_annotated`
in the filename — pass that exact path to `answer-acc --results`).

**Table 3 — Discovery answer accuracy vs. ground truth:**

```bash
python run.py answer-acc --dataset putnam \
    --results eval_dap_on_putnam_intermediate_1.0.json
# --judge gemini (default, used in the paper) | gpt-oss (open-source alternative)
```

**Table 5 — rewriting-strategy ablation:**

```bash
python run.py rewrite-ablation --dataset putnam --mode ours      # discover then rewrite
python run.py rewrite-ablation --dataset putnam --mode straight  # single-step rewrite
python run.py rewrite-ablation --dataset putnam --mode none      # no rewriting (Hard statement direct)
```

> The `none` and `straight` settings can produce *spurious proofs* (the prover
> sees the answer placeholder and cheats); the paper reports manually filtered
> numbers for these two columns.

## Datasets

See [DATASETS.md](DATASETS.md) for schema, sources, licenses, and the
reannotation / answer-leak fixes.

## Known issues / notes

- **Stochasticity:** results are not bit-reproducible due to LLM sampling.
- **MiniF2F-Hard counts:** this release uses the corrected **197**-Hard-Mode
  version (paper Table 1 / Table 3); the paper's Table 2 and SV-iteration
  ablation report the older 194. See DATASETS.md.
- **FIMO answer-leak fixes:** three FIMO statements were cleaned, so they differ
  slightly from the exact inputs that produced the paper's numbers (DATASETS.md).
- Out of scope for this repository: Easy Mode baseline re-evaluation, the
  self-verification iteration ablation, cross-model pairings (Qwen3 / Aristotle),
  and the manual failure-mode analysis (Table 4).

## License

Code is released under the [MIT License](LICENSE). Datasets remain under their
respective upstream licenses; see [DATASETS.md](DATASETS.md).

## Citation

If you find our work useful, please consider citing our paper.

```bibtex
@inproceedings{liu-etal-2026-discover,
    title = "Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4",
    author = "Liu, Chengwu  and
      Yin, Yichun  and
      Yuan, Ye  and
      Xie, Jiaxuan  and
      Li, Botao  and
      Li, Siqi  and
      Shen, Jianhao  and
      Xu, Yan  and
      Shang, Lifeng  and
      Zhang, Ming",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.3/",
    pages = "117--133",
    ISBN = "979-8-89176-390-6"
}
```
