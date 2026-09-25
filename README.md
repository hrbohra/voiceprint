# Voiceprint

**Extract the measurable voice of a speaker from a text corpus. Verify it as rules. Tune an LLM to it.**

Give Voiceprint a corpus (support threads, DMs, emails, a novel) and a speaker. It measures how that
speaker writes compared with their peers, turns the differences into plain-language rules, checks
each rule on conversations it never saw, and exports what you need to make an LLM write the same way:

```
out/
├── VOICE.md            human-readable voice guide, every rule with its evidence
├── voice.json          full profile: 155 features, contrasts, CIs, keyness, rules, situations
├── prompt_pack/        compact / standard / full system prompts
├── exemplars.jsonl     k-anonymous examples by situation, for few-shot retrieval
├── sft.jsonl           supervised fine-tuning pairs (context → reply)
├── dpo.jsonl           preference pairs (this voice vs a peer, same situation)
└── voiceScorer.ts      dependency-free "how on-voice is this?" scorer
```

It works on small corpora (a few hundred messages) and gets sharper with volume. The eval suite
measures exactly how much sharper.

## Why not just ask a frontier model to read the corpus?

You can: that is the `frontier-direct` tier. By default, though, Voiceprint measures first and lets
the LLM reason over a compressed statistical brief, for four reasons:

1. **Cost.** The brief plus a stratified sample is a few thousand tokens, not a whole corpus.
2. **Evidence.** Every claim carries a number, a confidence interval and an effect size. An LLM's
   impression carries none of these.
3. **Verification.** Rules are tested blind on held-out conversations, against a reference. Rules
   that are not true of the speaker, or not *distinctive*, are dropped.
4. **Privacy.** Anonymisation and k-anonymity run locally before anything leaves the machine.

## What it measures

| Family | Examples | How |
|---|---|---|
| Surface | length, rhythm, punctuation, emoji position, casing, contractions, hedges, boosters, textese, UK/US spelling, pronouns, readability, MTLD / HD-D | exact, no model |
| Syntax | POS profile, lexical density, Heylighen–Dewaele formality, tree depth, clauses, passive, imperative, modals, negation, tense | spaCy `en_core_web_trf` (GPU) |
| Affect | sentiment, 28 GoEmotions | RoBERTa classifiers |
| Dialogue acts | ask, answer, acknowledge, backchannel, … (conditioned on the previous turn) | DeBERTa trained on SILICONE |
| Social acts | greet, thank, apologise, offer, request, reassure, refuse, empathise, inform, close | zero-shot NLI per sentence plus strict cue phrases |
| Dialogue | reply length ratio, Language Style Matching, lexical echo, asking back, latency, bursts, openers, closers, act transitions | conversation structure |
| Situations | what the speaker is replying to | bge-large, HDBSCAN, class-based TF-IDF |
| Lexis | over- and under-used words and bigrams | Monroe et al. log-odds with an informative Dirichlet prior |

Confidence intervals bootstrap whole conversations, because turns in one chat are not independent.
Every model is pinned to a commit hash.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -e ".[gpu,anthropic,dev]"
python -m spacy download en_core_web_trf
voiceprint doctor                                    # GPU, spaCy, which providers have keys

voiceprint inspect chats.jsonl
voiceprint extract chats.jsonl -s "Host Name" -o out/            # with an LLM (ANTHROPIC_API_KEY)
voiceprint extract chats.jsonl -s "Host Name" -o out/ --no-llm   # fully local
voiceprint score out/voice.json "hey! yes that works 🙂"
```

Input formats: JSON Lines, CSV, WhatsApp export, Instagram/Messenger export, Customer Support on
Twitter, plain text, Markdown and Project Gutenberg books (narration and dialogue split).

## Providers and privacy

| Tier | What leaves the machine |
|---|---|
| `local-first` (default) | a statistical brief and a small sample of anonymised, k-anonymous turns |
| `hybrid` | every anonymised turn (for bulk LLM labelling) |
| `frontier-direct` | raw text. Refused unless the config acknowledges a zero-retention agreement |

Claude is the default: Opus 5 for induce, merge, judge and generate, and Haiku 4.5 only for bulk
labelling. OpenAI, Gemini and Ollama plug into the same interface. The router enforces the tier,
keeps a hard budget in GBP, caches every call, and refuses to send private data to a Gemini key that
may be free tier. Every published text passes a memorisation check: no output may contain a rare
8-gram from the corpus.

## Evaluation

```bash
voiceprint eval planted                      # synthetic corpus with known planted rules
voiceprint eval attribution corpus.csv       # can the features tell speakers apart on held-out turns?
voiceprint eval curve corpus.csv SPEAKER     # how results stabilise as data grows
voiceprint eval fidelity ...                 # does an LLM with the prompt pack write in the voice, same content?
```

Results and methodology: [`RESULTS.md`](RESULTS.md). How it was built, and why: [`BUILD_LOG.md`](BUILD_LOG.md).

## Licence

MIT. Model licences are their own. The optional formality ranker is CC BY-NC-SA and stays off
unless `research_models: true` is set.
