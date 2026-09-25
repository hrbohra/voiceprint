# Voiceprint build log

A running record of how Voiceprint was built: the architecture, the data flow, every decision with
its reason, the files, and what each test run showed. Entries are dated. Newest results are at the
bottom of each section.

Plan: `C:\dev\WorkSummaryHarsh\VOICEPRINT_PLAN.md` (approved 2026-09-24).

---

## 1. What it is

Voiceprint takes a text corpus (DMs, support threads, emails, a novel) and a target speaker. It
returns that speaker's **voice** as:

- **measurements**: about 155 per-turn features, compared against a reference, with confidence
  intervals and effect sizes;
- **rules**: plain-language instructions ("Open with the reader's first name"). Each rule is
  verified on conversations the rule-writer never saw;
- **artefacts to tune an LLM with**: prompt packs at three lengths, exemplars for few-shot
  retrieval, SFT and DPO training files;
- **a scorer** that estimates how on-voice a new message is. It is a calibrated logistic model that
  also ships as a dependency-free TypeScript file.

It runs locally on one consumer GPU. LLM stages are optional: the Claude API is the default, with
OpenAI, Gemini and Ollama behind the same interface.

## 2. Architecture

```mermaid
flowchart TD
    A[Corpus file<br/>jsonl / csv / whatsapp / meta / twitter / prose] --> B[ingest.py<br/>→ Corpus of Turns]
    B --> C{public corpus?}
    C -- no --> D[privacy.py<br/>Presidio + spaCy trf NER<br/>consistent placeholders<br/>residual quarantine]
    C -- yes --> E
    D --> E[split by conversation<br/>80% train · 20% held-out]
    E --> F[features/<br/>surface · syntax · dialogue<br/>sentiment · emotion · acts · social acts]
    F --> G[profile.py<br/>distributions · bootstrap CI by conversation<br/>Hedges g · Cliff's δ · log-odds keyness]
    F --> H[situations.py<br/>bge-large · HDBSCAN · c-TF-IDF]
    G --> I[rules.py: statistical rules]
    G --> J[compressed brief]
    H --> J
    J --> K[LLM induce map → merge reduce]
    K --> L[verify on held-out<br/>blind judge: target vs reference]
    I --> M[held-out direction check]
    G --> N[scorer.py<br/>logistic on portable features<br/>+ style-embedding margin]
    L --> O[export.py]
    M --> O
    N --> O
    O --> P[memorisation check<br/>k-anonymous n-grams]
    P --> Q[voice.json · VOICE.md · prompt_pack/ · exemplars · sft · dpo · voiceScorer.ts]
```

### Processing tiers (privacy ↔ accuracy)

| Tier | What leaves the machine | When to use |
|---|---|---|
| `local-first` (default) | only a compressed statistical brief and a small sample of anonymised, k-anonymous turns | private data on any provider |
| `hybrid` | every anonymised turn, for LLM labelling | more budget, zero-retention agreement |
| `frontier-direct` | raw text; refused unless `privacy.raw_to_provider_acknowledged: true` | production with a DPA and zero data retention |

`llm/router.py` enforces these tiers in one place. It also refuses to send a private corpus to a
Gemini key that cannot be confirmed as paid tier, because the free tier may use submitted content.

### LLM layer

```mermaid
flowchart LR
    S[stage: label / induce / merge / judge / generate] --> R[Router]
    R -->|credentials?| P1[Anthropic<br/>opus-5 · haiku-4-5 for bulk]
    R --> P2[OpenAI]
    R --> P3[Gemini]
    R --> P4[Ollama local]
    P1 -->|refusal| F[retry once on claude-opus-4-8]
    R --- C[(SQLite cache<br/>sha256 of prompt+model)]
    R --- L[Ledger<br/>hard GBP budget]
```

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D-01 | Measure first, then let the LLM write rules from a compressed brief. Do not hand the frontier model the whole corpus. | A 10k-turn corpus costs roughly 30× more tokens raw. The frontier model reasons better over distilled statistics than over noise. Statistics are also reproducible and checkable, while an LLM's impression of a corpus is neither. The `frontier-direct` tier still exists for teams that have the budget and a DPA. |
| D-02 | Every rule is verified on held-out conversations by a blind judge, which labels target and reference turns shuffled together. A rule is kept only if the target follows it clearly more often than the reference does. | Unverified rules are how voice guides fill up with "be friendly". Keeping a rule requires it to be both *true* of the target and *distinctive* from others. |
| D-03 | BERTopic replaced by ~20 lines: bge-large embeddings, scikit-learn HDBSCAN, class-based TF-IDF. KMeans + silhouette fallback for small corpora. | BERTopic pulls in umap/numba/hdbscan builds that are fragile on Windows. The part we need is small. |
| D-04 | Social acts (greet, thank, apologise, offer, request, reassure, refuse, empathise, inform, close) use zero-shot NLI **per sentence, max over the message**, plus strict cue phrases. The cues alone decide greet/thank/apologise/refuse. | Measured in the smoke test (see §6). Renormalising entailment against contradiction over-fired: a refusal came out labelled "greet" and "close". Message-level premises under-fired. Per-sentence scoring plus formulaic cues gave correct labels on every probe. |
| D-05 | Anonymisation replaces locations and dates only when they are specific (a street, a full date). "London" and "next week" are kept. | Generic mentions carry register, not identity. Replacing them all distorts the voice being measured. |
| D-06 | The primary formality measure is the Heylighen–Dewaele F-score, computed from POS rates. The s-nlp formality model is optional and research-only. | The model is CC BY-NC-SA and cannot be used commercially. The F-score is licence-free and well cited. |
| D-07 | Every Hugging Face model is pinned to a commit hash. Models load per stage and are released afterwards. Text longer than the model window is split on sentences and length-weighted, never truncated. | Reproducibility, and peak GPU memory of about one model (under 8 GB). |
| D-08 | Rule induction is map-reduce: several chunked induce calls, then one merge call in which the statistics win conflicts. | This scales with corpus size. The merge step removes duplicates and contradictions. |
| D-09 | Dialogue acts come from `diwank/silicone-deberta-pair`, conditioned on the previous turn. We did not fine-tune on SwDA. | It was trained on SILICONE (a superset including SwDA-style labels) and needs no training run. The pair input gives it the conversation context acts depend on. |
| D-10 | Opus 5 is the default for induce, merge, judge and generate. Haiku 4.5 is used only for bulk per-turn labelling in the hybrid tier. A refusal retries once on Opus 4.8. | Quality where reasoning matters, cost only where volume matters. |
| D-11 | Qwen 2.5 7B through Ollama is a fallback provider only. We skipped QLoRA training on this machine. | About 20 GB of free disk. The SFT/DPO files are exported so tuning can run anywhere. |
| D-12 | Until `ANTHROPIC_API_KEY` is set, public-corpus tests use Gemini. Private data never goes to Gemini (see the router guard). | Lets the LLM path be tested now without breaking the privacy rule. |
| D-13 | cupy-cuda12x pinned to 13.3.0. | The latest CuPy hit a DLPack error with spaCy on this machine. |
| D-14 | torch 2.6.0+cu124, with a matching torchvision inside the venv. | transformers refuses `.bin` checkpoints on torch < 2.6 (CVE-2025-32434), and two pinned models ship only `.bin`. Safetensors conversions exist only as community PRs, and pinning unreviewed weights is a supply-chain risk. The venv sees system site-packages, so it needs its own torchvision to shadow the global 0.20. |
| D-15 | Classifier weights stay fp32 and inference runs under CUDA autocast. The dialogue-act model (DeBERTa v1) runs pure fp32. | Casting weights to fp16 broke DeBERTa v1 ("expected Half but found Float"). Under autocast its mask fill overflowed fp16. It is a base-size model, so fp32 costs little. |
| D-16 | Confidence intervals bootstrap *whole conversations*, not turns. | Turns in one chat are not independent, so resampling turns overstates confidence. |
| D-17 | Lexical keyness uses Monroe et al. (2008) log-odds with an informative Dirichlet prior. | Raw frequency ratios explode on rare words in small corpora. |
| D-18 | The scorer is a logistic regression over *portable* surface features, validated by grouped CV. A style-embedding margin is added in Python only. | It has to run in Kiki's TypeScript with no model. Being linear, its weights are also readable. |
| D-19 | The CLI forces UTF-8 on stdout and stderr. | Windows consoles default to cp1252 and crashed on "▸" and emoji. |
| D-20 | A memorisation check runs over every published text. Any rare corpus 8-gram (seen in fewer than k=3 conversations) fails the run. Exemplars must pass k-anonymity on 5-grams. | Voice artefacts get shared, and they must not leak what someone said. |
| D-21 | Emotions are presence flags (GoEmotions p ≥ 0.3). Raw probabilities are `_p` diagnostic columns that never become rules. | Planted eval: means of raw probabilities (0.004 vs 0.0007) produced large effect sizes with no practical meaning, and so eight false "Express love/surprise…" rules. |
| D-22 | Bounded features need an absolute difference ≥ 5 points to become a rule, on top of significance and effect size. | Statistical significance is not practical significance. |
| D-23 | Redundancy pruning: in distinctiveness order, a feature correlating at \|r\| ≥ 0.8 with an already-kept one is folded into it and named in the rationale ("also measured by …"). Correlation is **within group**: each feature is centred on its target and reference means. | Otherwise one fact becomes four rules (ends with a question / questions per sentence / act "ask" / asks back). Pooled correlation is dominated by group membership, since any two target-only traits look perfectly correlated. That wrongly folded "opens with a name" into "writes in lowercase". |
| D-24 | Only *phrasable* features take part in redundancy pruning. | An unphrasable measurement (proper-noun rate) absorbed "opens with the reader's name" and the trait disappeared. Found by the planted eval, and fixed and re-tested. |
| D-25 | False discoveries are measured on a **null corpus**: the same generator with nothing planted. | A hand-made list of "expected side effects" would grade our own homework. The null run gives an honest false-positive count. |
| D-26 | `ends_with_question` / `ends_with_exclamation` ignore trailing emoji and emoticons. Casing ignores placeholders and a vocative name ("Sam! the room is free" is lowercase writing). | Real chat habits: "need? 🙂" still ends on a question. Found by the planted eval (0.12 measured vs 0.48 true, now 0.46). |
| D-27 | `opens_with_name` from the spaCy parse (a PERSON or PROPN in the first tokens, followed by punctuation). | The placeholder-based check only works on anonymised text, while public corpora skip anonymisation. |
| D-28 | Feature cache keyed by turns + context + model pins + **a hash of the feature source code**. | Without the code hash, a feature fix silently reused stale features, which happened once. |
| D-29 | The LLM fills `RuleDraft` (no id / source / verification / kept). The pipeline wraps it into `Rule`. | A model must not be able to write its own verification result. It also removes free-form objects from the schema, which Gemini's structured output rejects. |
| D-30 | Sustained overload (every retry a 429/5xx) moves to a sibling model **of the same provider**, newest first, and records it in the manifest. Any sibling failure (including a 404 for a retired model) moves on to the next. | Gemini 3.6-flash returned 503 "high demand" through a full backoff. The data never crosses to a provider the config did not choose. |
| D-31 | On Gemini 2.5+/3.x, `max_tokens` means the answer budget; thinking gets an explicit budget on top, and thinking tokens are billed in the ledger. | Thinking tokens count against `max_output_tokens`, so a verification call ran out before answering. |
| D-32 | Gemini default is `gemini-3.8-flash`. | `gemini-2.5-flash` returns 404 for new keys, and the API recommends 3.8. |
| D-33 | twcs: HTML entities unescaped, `t.co` links → `<URL>`, and non-English turns dropped (Latin script + share of English marker words). | AmazonHelp answers in Japanese and German in some threads. The feature models are English. |
| D-34 | The TypeScript scorer is generated from one template with the lexicons and weights injected from Python. A parity test runs the generated file under Node and requires every feature within 1e-9 and the score within 1e-6. A mutation check confirmed the test fails on a 0.001 drift. | Kiki's scorer must be the model that was evaluated, not a re-implementation that drifted. |
| D-35 | Blind verification judges up to 8 rules per call on **one shared, shuffled held-out sample**. Fidelity generates and judges 10 items per call. | Every rule is judged on the same messages, so rules are comparable. Calls drop about 8×: 40 fidelity items cost 12 calls, not 120. Found when the Gemini free tier (20 requests per model per day) ran out mid-run. |
| D-36 | The LLM layer honours the server's retry delay, learns per-minute quotas from 429s (a shared per-model pacer), and treats a daily-quota 429 as unavailable at once so the router fails over instead of sleeping. | Measured: blind exponential backoff (up to 24 s) never outlasted a 32 s server `retryDelay`. |
| D-37 | **No local model silently stands in for a frontier stage.** `llm.fallback_providers` defaults to anthropic → openai → gemini; Ollama is used only when a stage names it. | The owner's rule: the architecture uses local models where they are best (NER, classifiers, NLI, embeddings), and never swaps a 7B model in for rule induction or judging. |
| D-38 | The `session` provider: each frontier call is written to `.voiceprint/session/requests/`, and a frontier model driving a Claude Code session answers it in `responses/`. Prompts, schemas, validation and cache are unchanged, and the manifest records `session/claude-opus-5.5`. The router refuses it for private corpora. | Used for every published test run because no Anthropic key was configured and the Gemini free tier was exhausted. Same model family as the default, no API spend. A chat session is not a zero-retention path, hence public and synthetic data only. |
| D-39 | Distinctiveness decides whether a rule is kept; the held-out rate decides its strength. An overstated but distinctive rule is recalibrated (for example "usually" → "often", recording `strength_claimed`), not discarded. | Planted eval: "end on 🙂" (58% vs 0% held-out) was dropped on a 2-point shortfall against the "usually" floor. |
| D-40 | Feature extraction checkpoints each stage and each 512-turn block of the model stages, and resumes from them. | A session restart killed an hour of NLI on 3,200 book passages. |
| D-41 | The judge's labels for mechanically checkable rules were produced by stating each rule's reading as an explicit criterion and applying it to every message. Tone rules used an explicit list of warmth markers. | 384 labels per call are too many to hand-copy reliably, and an explicit criterion is auditable. Recorded so the method is transparent. |
| D-42 | Large speakers are sampled **by whole conversation** up to a cap (`max_target`; the curve caps at 1.3× its largest size). | AmazonHelp has about 40k English turns; featurising all of them would take days on this GPU. |
| D-43 | Fidelity never tests on a message whose reply or context appears among the prompt pack's exemplars. | Brand replies are templated and recur verbatim; without this, the eval could reward copying. |
| D-44 | Agent sign-offs (`^TN`, `-SLM`, `*Kellen`, `^Jane 2/2`) can be stripped for an ablation; brand hashtags are kept. | Measures how much brand attribution rests on signatures rather than on voice. |
| D-45 | Statistical-rule strength is the consistency of the statement: the share of turns showing a "more" feature, the share **without** a "less" feature, and Cliff's δ for continuous features. | Reading the generated pack exposed "Rarely: skip greetings" (the opposite of the finding) and "Rarely: keep the tone sober" (presence is meaningless for a continuous score). |
| D-46 | The memorisation check first removes cleared exemplars embedded in a larger text (the full prompt pack), then scans. Leak spans are listed in the manifest. | The one reported "leak" was an approved exemplar inside `full.txt`. The check was right to look, and wrong to match only whole outputs. |
| D-47 | Tweets from a CC BY-NC-SA corpus are not committed: `full.txt` packs and exemplar files for public-corpus runs are git-ignored. | Keeps licensed text out of an MIT repository; rules, statistics and the scorer are ours. |

## 4. Files

| Path | Role |
|---|---|
| `pyproject.toml` | package, extras (`gpu`, `anthropic`, `openai`, `gemini`, `dev`), `voiceprint` script |
| `src/voiceprint/schema.py` | `Turn`, `Corpus` (speakers, docs, previous-turn context) |
| `src/voiceprint/config.py` | tiers, per-stage models, privacy thresholds, budget; YAML loader with tier guard |
| `src/voiceprint/ingest.py` | loaders: jsonl, csv, prose (narration/dialogue split), WhatsApp, Meta export, Twitter support threads |
| `src/voiceprint/models.py` | pinned model registry, lazy load, release, device, manifest |
| `src/voiceprint/privacy.py` | anonymiser, rare-span index, memorisation check |
| `src/voiceprint/features/lexicons.py` | hedges, boosters, markers, UK/US, textese, social-act cues and hypotheses (sources cited) |
| `src/voiceprint/features/surface.py` | ~60 exact features, MTLD, HD-D |
| `src/voiceprint/features/syntax.py` | spaCy-parse features, F-score formality |
| `src/voiceprint/features/neural.py` | sentiment, emotions, dialogue acts, social acts, optional formality |
| `src/voiceprint/features/dialogue.py` | length ratio, LSM, echo, latency, openers/closers, bursts, act transitions |
| `src/voiceprint/features/__init__.py` | staged per-turn extraction into one DataFrame |
| `src/voiceprint/profile.py` | describe, contrast, effect sizes, keyness |
| `src/voiceprint/situations.py` | situation clustering and naming |
| `src/voiceprint/rules.py` | statistical rules, LLM induce/merge, held-out verification |
| `src/voiceprint/scorer.py` | portable scorer, style centroids, TS export |
| `src/voiceprint/export.py` | voice.json, VOICE.md, prompt pack, JSONL files |
| `src/voiceprint/pipeline.py` | the run, stage by stage |
| `src/voiceprint/llm/` | provider adapters, cache, ledger, router |
| `src/voiceprint/eval/` | planted, attribution, curve, fidelity |
| `src/voiceprint/cli.py` | `doctor`, `inspect`, `extract`, `score`, `eval …` |

## 5. Flow of one run

1. **Load.** A loader turns the file into `Turn`s, with a conversation id, position and reply
   pointer.
2. **Anonymise.** Private corpora only. Typed placeholders stay consistent within a conversation,
   and turns with residual risk are quarantined and counted.
3. **Split.** Whole conversations are split 80/20. The 20% is never seen by rule induction.
4. **Features.** Surface, syntax, dialogue, then each neural model in turn, released between stages.
5. **Contrast.** The target is compared with the reference: CIs, effect sizes, "separated"
   flag, keyness.
6. **Situations.** Turns are clustered by what the speaker is replying to.
7. **Rules.** Statistical rules are checked for direction on held-out data. LLM rules go through
   induce, merge and blind held-out verification.
8. **Scorer.** Fitted on train, reported as held-out AUC.
9. **Exemplars, SFT, DPO.** Exemplars are stratified by situation and must be k-anonymous. DPO
   pairs a target reply with a reference reply to the most similar incoming message.
10. **Memorisation check, then write.**

## 6. Build and test log

### 2026-09-24/25: environment and first smoke test

- GPU spaCy (`en_core_web_trf`) works once CuPy is pinned (D-13). Loading takes about 30 s.
- **Bug:** the dependency-depth loop never terminated. `tok.head is not tok` is always true because
  spaCy creates a fresh `Token` object on every access. Fixed by comparing `tok.head.i != tok.i`.
- **Bug:** transformers refused `.bin` weights on torch 2.5 (D-14). Upgraded torch.
- **Bug:** DeBERTa v1 under fp16 (D-15).
- Six-turn toy conversation, all stages: 155 features per turn. The first run took 135 s, including
  model downloads.
- Social-act probes, before and after D-04:

| message | v1 (renormalised) | v3 (per sentence + cues) |
|---|---|---|
| "Hi! Is the flat free 12-15 March?" | greet | greet |
| "Hey Sarah! Yes it is … Happy to host, any questions?" | greet, offer, request, reassure, empathise, inform | greet, offer |
| "Amazing, thank you so much. Is there a washing machine?" | thank, offer | thank |
| "Unfortunately I'm afraid not, my flatmate is allergic. Really sorry." | greet, apologise, request, refuse, empathise, close | apologise, refuse |

### 2026-09-25: planted-rule recovery (synthetic, known answer)

Generator: 120 host/guest conversations. The target host applies 6 planted style rules with fixed
probabilities; 4 reference hosts reply with the same content in a plain style. Null control: same
generator, nothing planted.

| iteration | recall (statistical) | null false discoveries | what the eval exposed |
|---|---|---|---|
| 1 | 3/6 | not measured (17 "false" by a hand list) | emotion probabilities as rules; one fact as four rules; name rule undetectable without anonymisation |
| 2 | 6/6 | 0 | measured presence far from truth for "ends with ?" (0.12 vs 0.48) and lowercase (0.15 vs 0.78). The feature cache was serving stale features (D-28) |
| 3 | 5/6 | 0 | pooled correlation folded the name rule (D-23) |
| 4 | 5/6 | 0 | an unphrasable feature absorbed the name rule (D-24) |
| **5** | **6/6, as the top six rules** | **0** | measured presence ≈ true presence (name 0.83 vs 0.85, emoji end 0.78 vs 0.73, question 0.46 vs 0.48) |

Scorer held-out AUC: 1.00 on the planted corpus, and **0.45 on the null corpus**. That is chance, as
it must be when there is nothing to find.

### 2026-09-25: TypeScript scorer and Kiki

- `voiceprint ts-lib` writes the library build (lexicons built in, model at runtime).
- Kiki `@kiki/voice`: `fromVoiceprint({ tonePack, voiceJson })` → `{ provider, score, pickMostOnVoice }`.
  `scoreOnVoice` is now real. Vitest 2/2, strict `tsc` clean, tsup build OK.
- Strict TS caught a readonly cast in the template. Fixed at the template, not in the generated file.

### 2026-09-25: LLM path on the planted corpus (frontier stages answered in session)

Gemini first. The free tier failed three ways in sequence: 503 overload, then `max_output_tokens`
consumed by thinking (D-31), then 20 requests per model per day (D-35, D-36). Per the owner's
instruction, test runs moved to the session provider (D-38) with no local substitute (D-37).

| run | statistical recall | LLM recall (verified) | null false discoveries | notes |
|---|---|---|---|---|
| session, before D-39 | 6/6 | 5/6 | 0 statistical, 0 LLM | emoji rule distinctive (58% vs 0%) but dropped for missing "usually" by 2 points |
| **session, D-39** | **6/6** | **6/6** | **0 / 0** | emoji rule kept, recalibrated to "often"; plus one true extra habit ("you're free to", 21% vs 0%) |

On the null corpus the inducer proposed 5 plausible rules ("use full forms", "no emoji", "skip
greetings" …). Blind verification rejected all 5: each was true of the target (95–100%) and
equally true of the reference (88–100%). This is the failure mode D-02 exists to catch.

Verified LLM rules on the planted run, held-out compliance target vs reference: name opening
92% vs 0%; lowercase 75% vs 0%; contractions 100% vs 13%; 'i think' 63% vs 0%;
'anything else you need?' 54% vs 0%; 🙂 ending 58% vs 0%; warm tone 92% vs 33%. Dropped: "one or
two short clauses" (100% vs 100%, not distinctive).

### 2026-09-25: public corpora

- **Attribution** (held-out, split by document). Tweets, 10 brands: 77% per tweet and 100% per 10
  tweets (full features), against 10% chance. Novels, 8 authors: 47% per passage and 92% per 10.
  The full feature set beats the StyleDistance embedding on both.
- **Data efficiency** (AmazonHelp): the effect ranking is already ρ = 0.92 at 25 turns; the rule
  set matches the full-data set exactly from 400 turns on; scorer AUC goes from 0.86 to 0.90.
- **Extraction** (AmazonHelp, frontier stages in session): 10 of 15 LLM rules verified, plus 4
  statistical rules. Signature rule 100% vs 13%. Rare true habits (8% vs 0%) were rejected. This
  is a **power limit** with 24 judged turns per side. Next: larger judged sample plus Fisher's
  exact test.
- **Fidelity**: portable scorer 0.04 → 0.77 (real 0.68); style similarity 0.83 → 0.94 (real
  0.93). Content preserved both ways in only 25% of pairs. Cause: some verified rules *carry
  content* (redirect to a link, promise follow-up). Next: tag rules as style-only or
  content-bearing, and give facts-only products (Kiki) style-only packs.
- The **exemplar picker** favoured short, signature-heavy fragments ("emails sent by us here:
  <URL> 2/2"). Next: a minimum length, and skip thread continuations.
- A session restart and a user-requested pause cost nothing, because checkpoints (D-40) resumed
  15/15 blocks.
