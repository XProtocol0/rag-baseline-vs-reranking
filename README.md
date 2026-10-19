# rag-baseline-vs-reranking

Two RAG pipelines over the **FIA 2026 Formula 1 Technical Regulations** (Section C, Issue 20, 254 pages), compared on the same 1,032 clause-aware chunks:

1. **Baseline:** a bi-encoder (`all-MiniLM-L6-v2`) with FAISS retrieves the top 10 chunks, and the top 3 go to the LLM.
2. **Re-ranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores those same 10 chunks, and its top 3 go to the LLM.

Both pipelines generate answers with DeepSeek (`deepseek-flash`) using an identical prompt.

Everything (code, results and the report) is in **[rag_reranking.ipynb](rag_reranking.ipynb)**.

## Results

| Query | Failure type | Gold-clause rank (bi-encoder → cross-encoder) | Baseline answer |
|---|---|---|---|
| Is a launch control that detects the start lights allowed? | Negation | C8.3.1: #4 → #1 | "Not enough information" |
| Are rotary engines allowed? | Keyword trap | C5.1.1: #8 → #1 | "Not enough information" |
| Must every part be shown legal by physical inspection? | Exception | C1.5: #6 → #2 | Right verdict, but it missed the "except for safety features" clause |
| Can the driver cancel a driver default during the start? | Constraint | C8.2.4: #5 → #1 | "Not enough information" |
| Can a system cheat the wing flex tests? | Negation | C3.18.1: #6 → #1 | "Not enough information" |
| Can the wheel nut hold other parts? | Buried chunk | C10.9.1: #7 → #3 | "Not enough information" |
| How much can the battery charge swing per lap? | Buried chunk | C5.2.9: #4 → #3 | Quoted the wrong limit |

The "gold clause" is the regulation clause that actually answers each question. The re-ranked pipeline answered all 7 correctly, each time citing that clause.

Re-ranking costs about **+31 ms per query** on CPU, about 2% of end-to-end latency. Cross-encoding all 1,032 chunks instead would take **2.4 s per query**.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # then put your DeepSeek API key in .env
.venv/bin/jupyter notebook rag_reranking.ipynb   # then Kernel → Restart & Run All
```

To run it headless:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace rag_reranking.ipynb
```

The first run downloads both models (~180 MB) from Hugging Face. A full run takes about 2 minutes on a laptop CPU.
