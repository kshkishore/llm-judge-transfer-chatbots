# Do validated LLM judges transfer to real chatbot replies?

Data, code and complete model-call logs for the manuscript

> **Do Validated LLM Judges Transfer to Real Chatbot Replies? Judge Disagreement, Rule-Based Checks and Implications for Evaluating Conversational Agents**
> (submitted to the *International Journal of Human–Computer Interaction*)

Two open-source chatbots answered 160 everyday requests. Two small open-weight LLM judges then scored the replies. The same repository also documents how the judges were validated beforehand, on a contrast set of designed appropriate and inappropriate replies. That validation is reported in a companion manuscript.

Everything ran locally on a two-core CPU with four-bit GGUF models via llama.cpp. Decoding was deterministic, and every judge output was grammar-constrained and logged.

| Role | Model |
|---|---|
| Judge J1 | Qwen3-VL-4B-Instruct |
| Judge J2 | Phi-4-mini-instruct (text only) |
| Chatbot G1 | Qwen3-VL-2B-Instruct |
| Chatbot G2 | SmolVLM2-2.2B-Instruct |

## Repository layout

```
study/                     code, items and chart images
study/results/             logs of every model call and derived tables
development_logs/          judgments made while the instrument was developed (48 items)
user_study_kit/            materials for the planned user study (no participant data)
```

### The chatbot study (main subject of the manuscript)

| File | Content |
|---|---|
| `study/results/gen_qwen3vl-2b.jsonl`, `gen_smolvlm2-2.2b.jsonl` | The two chatbots' replies to all 160 requests |
| `study/results/genjudge_j1.jsonl`, `genjudge_j2.jsonl` | Rubric judgments of every reply (written diagnosis + six scores) by J1 and J2 |
| `study/results/gen_table_v2.csv` | One row per reply, giving the scores of both judges, the format and chart-fact rules, metrics and length |
| `study/results/metrics_gen.json` | Automatic metrics for the replies |
| `study/results/summary_v2.json` (`exp5`) | Chatbot comparisons, judge agreement, criterion checks |
| `study/results/qual_check_G1_O_le2.md` | Manual check of 14 sampled misreading flags |
| `study/extra_analysis.py` → `study/results/extra_results.json` | Additional analyses in the manuscript, including scores by format compliance, length correlations within each chatbot, diagnosis terms and misreading flags by category |
| `study/make_fig.py` | Figure 1 (J1 vs. J2 scores). Set `TERMES_DIR` to a folder containing the TeX Gyre Termes fonts |

### The prior validation (companion manuscript)

- **Items:** `study/golden_dataset_v2.json` holds all 160 items in ten categories. Each item has the user request, a six-field specification (Context, Objective, Style, Tone, Audience, Response format), a reference reply and three designed replies:
  - A, appropriate;
  - B, a misreading;
  - C, the wrong register.

  Chart items carry the data table and an answer key, and the charts are in `study/images/`.
- **Building the items:** `study/build_v2.py` assembles and validates the dataset. Items are defined in `study/items_*.py`, `golden_dataset.json` (32 development items) and `golden_dataset_ext.json`.
- **Judging code:** `study/pipeline.py` holds the models, prompt templates, output grammars and chains. `run.py`, `run2.py`, `robust.py` and `phases.sh` run the judging phases.
- **Metrics:** `study/metrics*.py` compute the 17 automatic metrics.
- **Statistics:** `study/analyze_v2.py` computes all statistics and writes `results/summary_v2.json`.
- **Logs:**
  - `contrast_j1.jsonl` and `contrast_j2.jsonl`: the three prompt conditions;
  - `pairwise_j1.jsonl`: side-by-side judgments in both orders;
  - `robust_j1.jsonl`: reworded rubric and sampled runs.
- **Model checksums:** `study/results/sha256.json`.
- **Re-running development calls:** `study/check_reuse.py` re-runs a sample of the development calls carried over from `development_logs/` and confirms that each reproduces its logged output exactly.

## Re-running

1. Use Python 3.13 with:
   - `llama-cpp-python==0.3.36`, `langchain-core==1.7.6`, `pydantic==2.13.5`
   - `sacrebleu==2.6.0`, `rouge-score==0.1.2`, `bert-score==0.3.13`, `nltk` (with WordNet), `textstat`
   - `transformers==5.22.2`, `torch==2.15.0` (CPU)
   - `matplotlib`, `scipy`, `pandas`, `scikit-learn`, `krippendorff`
2. Download the four-bit GGUF files of the four models, check them against `study/results/sha256.json`, and set `MODELS_DIR` to their folder.
3. In `study/`, run:
   ```
   python build_v2.py
   ./phases.sh gen j1_gen j2_gen                  # chatbot replies and their judgments
   ./phases.sh j1_text j1_mm j1_hol j2 pair rob   # prior validation on the contrast set
   python metrics_v2.py contrast
   python metrics_v2.py gen
   python analyze_v2.py
   python extra_analysis.py
   python make_fig.py
   ```

Each phase skips calls already in its log, so runs can be resumed and the shipped logs are reused. Decoding is greedy with seed 20261109. The sampled robustness runs instead use temperature 0.7 with seeds 101, 202 and 303.

## User study kit

`user_study_kit/` contains materials for a planned study in which people rate the chatbot replies. It includes:

- 80 blinded replies with chart images;
- the hidden key linking each reply to its chatbot and scores;
- the protocol and hypotheses;
- the survey questions;
- an analysis script.

It contains **no participant data**.

## Content note

Some designed C replies in the contrast set deliberately contain condescension or a cultural stereotype, as examples of inappropriate replies. Do not use them as generation targets.

## Licence

The data (items, replies, logs) are licensed under [CC BY 4.0](LICENSE-DATA). The code is licensed under the [MIT licence](LICENSE).

## Citation

See `CITATION.cff`. Until publication, please cite the manuscript as under review.
