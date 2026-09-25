# Results

Every number here is reproducible from this repository. Corpora are not redistributed; the fetch
steps are listed under each eval. Model revisions are pinned (`models.py`). Frontier LLM stages in
these runs were answered by Claude Opus 5.5 through the `session` provider (see
[BUILD_LOG.md](BUILD_LOG.md), D-38), not through the API. The prompts, schemas and validation are
identical, and the manifest of every run records it.

## 1. Planted-rule recovery (known answer)

`voiceprint eval planted` (statistical only) · `voiceprint eval planted --config configs/session.yaml` (with LLM stages)

A generator writes 120 host/guest conversations. The target host applies six planted style rules,
each with a fixed probability; four reference hosts answer the same questions in a plain style. A
**null corpus** from the same generator plants nothing, so anything found there is a false discovery.

| | statistical rules | LLM rules (blind-verified) |
|---|---|---|
| planted rules recovered | **6 / 6**, the top six rules | **6 / 6** |
| false discoveries on the null corpus | **0** | **0** (5 proposed, 5 rejected) |
| scorer held-out AUC, planted / null | 1.00 / **0.45** (chance, as it should be) | |

Measured presence against the truth: name opening 0.83 vs 0.85, emoji ending 0.78 vs 0.73,
question ending 0.46 vs 0.48, lowercase 0.85 vs 0.78.

LLM rules after blind held-out verification (target vs reference compliance):

| rule (as induced) | target | reference | kept |
|---|---|---|---|
| Open by addressing the guest by first name followed by "!" | 92% | 0% | ✓ |
| Write in lowercase, capitalising only the guest's name | 75% | 0% | ✓ |
| Use contractions wherever possible | 100% | 13% | ✓ |
| Soften the answer with "i think" | 63% | 0% | ✓ |
| Offer more help with "anything else you need?" | 54% | 0% | ✓ |
| End on a single 🙂 (claimed "usually", recalibrated to "often") | 58% | 0% | ✓ |
| Grant permission with "you're free to / welcome to" | 21% | 0% | ✓ |
| Keep the tone warm and upbeat | 92% | 33% | ✓ |
| Answer in one or two short clauses | 100% | 100% | ✗ not distinctive |

On the null corpus the inducer proposed "use full forms", "no emoji", "skip greetings" and two
more. All were true of the target and equally true of the reference, and all were rejected.

## 2. Authorship attribution on held-out text

`python scripts/public_evals.py books` · `python scripts/public_evals.py twcs`

If the features capture voice, they should tell writers apart on text never used for fitting.
Held-out material is split by document: whole conversations for tweets, and 25-paragraph blocks
for books. Three classifiers run on the same split: the **surface** features only (what the
TypeScript scorer sees), the **full** feature set, and the **style** embedding alone
(StyleDistance, nearest centroid, no training).

| corpus | writers | chance | surface: 1 text / 10 texts | full: 1 / 10 | style embedding: 1 / 10 |
|---|---|---|---|---|---|
| Customer Support on Twitter (brand replies) | 10 brands | 10% | 64% / 95% | **77% / 100%** | 61% / 93% |
| Project Gutenberg novels (narration) | 8 authors | 12.5% | 33% / 76% | **47% / 92%** | 31% / 64% |

The full feature set beats the purpose-built style embedding on both corpora, and the cheap
portable subset is close behind on tweets. A voice shows over several messages: ten texts are
enough to identify the writer almost every time.

Caveats. Book test passages come from the same novels as training passages (disjoint blocks), not
from unseen novels. Many brands sign replies with agent initials (`^TN`); that is part of brand
voice, but it makes brand attribution easier than attribution in general.

Data: twcs from the Hugging Face mirror `SunidhiSriram/twcs` @ `b03fa0a7` of ThoughtVector's
*Customer Support on Twitter* (CC BY-NC-SA 4.0), first 700k rows, English turns only. Books: 23
Project Gutenberg texts by Austen, Dickens, Twain, Doyle, Wilde, C. Brontë, Hardy and Wells.

## 3. How much data does a voice need?

`python scripts/public_evals.py curve` (AmazonHelp against 9 other brands, 415 held-out AmazonHelp turns)

Features are computed once. The contrast, the statistical rules and the scorer are then re-fitted
on growing random subsets of the target's training turns, and compared with the full-data result.

| target turns | rank agreement of effect sizes with full data (Spearman ρ) | same rule set as full data (Jaccard) | scorer AUC on held-out turns |
|---|---|---|---|
| 25 | 0.92 | 0.23 | 0.86 |
| 50 | 0.94 | 0.36 | 0.88 |
| 100 | 0.95 | 0.64 | 0.87 |
| 200 | 0.96 | 0.70 | 0.89 |
| 400 | 0.98 | **1.00** | 0.89 |
| 800 | 0.99 | 1.00 | 0.90 |
| 1,665 | 1.00 | 1.00 | 0.90 |

The measured profile is stable from 25 messages (ρ = 0.92). The rule set needs more: it matches
the full-data rules exactly from 400 messages on. Below that, it is a mix of the true rules and
chance-level ones that the held-out check is there to catch. For Kiki's roughly 10,000
conversations this sits well inside the stable region: small corpora work, and volume buys
certainty about the rules.

## 4. A full extraction: AmazonHelp

`python scripts/public_evals.py extract` → [`results/voice_AmazonHelp/`](results/voice_AmazonHelp/) (`VOICE.md`, `voice.json`, prompt pack, scorer)

2,000 AmazonHelp turns (sampled by conversation; 376 held out) against 9 other brands. The
frontier stages were induction over 6 chunks of 60 turns, one merge, and blind verification on
24 + 24 held-out turns. Verification kept 10 of 15 merged LLM rules, plus 4 statistical rules.
Scorer held-out AUC: 0.84 (portable surface model), 0.89 (with style embedding). The
memorisation check found no leaks.

| verified rule | AmazonHelp | other brands |
|---|---|---|
| End with " ^" + agent initials | 100% | 13% |
| Skip "Hi/Hey"; begin with the apology or answer | 92% | 75% |
| Move the issue to a private channel via a link | 33% | 13% |
| Speak as "we" and promise follow-up | 29% | 13% |
| Open with an apology for the specific problem | 21% | 0% |
| Formal service formulae ("Kindly", "revert", "correspondence") | 17% | 0% |

Rejected though true: the privacy warning (8% vs 0%), "I understand your concern" (8% vs 0%),
and tracking/carrier questions (8% vs 0%). This is a **power limit**, not a false rule. With 24
judged turns per side, a habit in under about 1 message in 6 cannot clear the 15-point margin.
Next step: a larger judged sample and an exact test (Fisher) in place of a fixed margin.

## 5. Does an LLM given the pack write in the voice, and say the same thing?

`python scripts/public_evals.py fidelity` (40 held-out AmazonHelp conversations, none shown to the model as exemplars)

The same generator replies twice to each incoming tweet: **baseline** ("reply helpfully and
naturally") and **voiced** (the standard prompt pack).

| | baseline | voiced | real AmazonHelp reply |
|---|---|---|---|
| voice scorer (portable, model-free) | 0.04 | **0.77** | 0.68 |
| style-embedding similarity (StyleDistance) | 0.83 | **0.94** | 0.93 |
| content preserved (NLI entailment both ways) | | 25% | |
| voiced reply entailed by baseline | | 50% | |

The pack moves generation to the measured voice: level with the real replies on style similarity,
and slightly past them on the scorer (the pack applies the habits more consistently than people
do). The pairwise style judge preferred the voiced reply in 40/40 pairs. The judge was the same
session model that wrote both replies, applying an explicit marker-count criterion, so this is
**not an independent judgement** and is reported only for completeness.

Content preservation is the weak spot, and the reason is instructive. Some verified rules
**carry content**: "move the issue to a private channel via a link" and "promise follow-up"
change *what* is said, not only *how*. For a product like Kiki, where the content must come
only from facts, those rules must be separated out. Next step: tag each rule as style-only or
content-bearing (a rule is content-bearing when applying it changes NLI entailment against the
unvoiced reply), and build facts-only packs from style-only rules.

Frontier stages (generation and judging) in this section were answered by Claude Opus 5.5 in a
Claude Code session through the `session` provider.

<!-- section 6 (signature ablation) is filled in when it completes -->
