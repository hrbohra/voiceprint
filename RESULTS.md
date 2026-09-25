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

<!-- sections 2–5 are filled in as the public-corpus evals complete -->
