# Voice of target_host

Corpus: null · 117 target turns · 235 reference turns · tier `local-first`

## Rules

| # | Rule | When | Strength | Evidence |
|---|---|---|---|---|

<details><summary>5 candidate rules failed verification</summary>

- Use full, uncontracted forms (it is, you are, there is, I will). ({'target_compliance': 1.0, 'reference_compliance': 1.0, 'n_target': 21, 'n_reference': 24, 'judge': 'session/claude-opus-5.5'})
- Answer in one plain declarative sentence with standard capitalisation and a full stop. ({'target_compliance': 0.952, 'reference_compliance': 0.875, 'n_target': 21, 'n_reference': 24, 'judge': 'session/claude-opus-5.5'})
- Skip greetings and sign-offs; start with the answer. ({'target_compliance': 1.0, 'reference_compliance': 1.0, 'n_target': 21, 'n_reference': 24, 'judge': 'session/claude-opus-5.5'})
- Grant permission with 'you are free to' or 'you are welcome to'. ({'target_compliance': 0.143, 'reference_compliance': 0.083, 'n_target': 21, 'n_reference': 24, 'judge': 'session/claude-opus-5.5'})
- Do not use emoji. ({'target_compliance': 1.0, 'reference_compliance': 1.0, 'n_target': 21, 'n_reference': 24, 'judge': 'session/claude-opus-5.5'})

</details>

## Most distinctive measurements

| Feature | Target | Reference | Hedges g | Cliff's δ |
|---|---|---|---|---|
| flesch_kincaid_grade | 3.36 [3.05, 3.71] | 3.83 | -0.25 | -0.16 |
| flesch_reading_ease | 93.2 [91, 95.3] | 90.1 | 0.25 | 0.15 |
| length_ratio | 1.34 [1.29, 1.39] | 1.42 | -0.23 | -0.15 |
| long_word_rate | 0.121 [0.112, 0.132] | 0.136 | -0.19 | -0.09 |
| mean_word_length | 3.71 [3.65, 3.77] | 3.78 | -0.18 | -0.10 |
| emotion_embarrassment_p | 0.00028 [0.00026, 0.00031] | 0.00032 | -0.19 | -0.08 |
| emotion_sadness_p | 0.001 [0.00091, 0.00108] | 0.0011 | -0.18 | -0.07 |
| chars | 59 [57.5, 60.2] | 60.3 | -0.15 | -0.07 |
| emotion_fear_p | 0.00055 [0.0005, 0.00059] | 0.00059 | -0.16 | -0.04 |
| questions_per_sentence | 0 [0, 0] | 0 | 0.00 | 0.00 |
| line_breaks | 0 [0, 0] | 0 | 0.00 | 0.00 |
| is_fragment | 0 [0, 0] | 0 | 0.00 | 0.00 |
| exclamations_per_sentence | 0.208 [0.127, 0.299] | 0.164 | 0.12 | 0.04 |
| ellipses_per_sentence | 0 [0, 0] | 0 | 0.00 | 0.00 |
| multi_exclamation | 0 [0, 0] | 0 | 0.00 | 0.00 |
| multi_question | 0 [0, 0] | 0 | 0.00 | 0.00 |
| commas_per_sentence | 0.365 [0.278, 0.479] | 0.286 | 0.17 | 0.08 |
| dashes_per_sentence | 0 [0, 0] | 0 | 0.00 | 0.00 |
| ends_with_question | 0 [0, 0] | 0 | 0.00 | 0.00 |
| ends_with_exclamation | 0.208 [0.126, 0.286] | 0.164 | 0.12 | 0.04 |
| ends_without_punctuation | 0 [0, 0] | 0 | 0.00 | 0.00 |
| words | 12.5 [12.2, 12.8] | 12.7 | -0.07 | 0.01 |
| emoji_count | 0 [0, 0] | 0 | 0.00 | 0.00 |
| quotes | 0 [0, 0] | 0 | 0.00 | 0.00 |
| has_emoji | 0 [0, 0] | 0 | 0.00 | 0.00 |

## Dialogue habits

- Starts 0% of conversations, ends 100%.
- Sends multiple messages in a row 0% of the time (mean burst 1.00).
- Top openers: `you can` ×7, `the wifi` ×7, `it is` ×6, `that is` ×4, `yes the` ×4, `the cafe` ×3
- Top closers: `to stay` ×7, `double glazed` ×5, `the bathroom` ×5, `the hallway` ×4, `for you` ×3, `host you` ×3

## Situations

| Situation | Turns | Keywords |
|---|---|---|
| wifi and remote work question | 13 | what's, wifi, like, need, work, flat |
| laundry and towels question | 11 | washing, machine, use, spare, towel |
| availability check for dates | 11 | room, free |
| bringing a bike | 10 | mind, bring, bike |
| late check-in request | 10 | hello, check, bit, late |
| leaving bags after check-out | 8 | leave, bags, check-out |
| train delay notice | 8 | sorry, train, delayed, i'll |
| thanks after a stay | 7 | much, lovely |
| noise at night concern | 6 | quiet, night, i'm, light, sleeper |
| breakfast recommendation request | 6 | recommend, somewhere, breakfast, nearby |
| neighbourhood walkability question | 6 | neighbourhood, easy, walk |

## Privacy

- 704 turns in, 0 quarantined, replacements: {}.
- Exemplars are k-anonymous (k=3); memorisation check: 0 leaks.

## Run

- Models: cuda · spaCy en_core_web_trf-3.8.0
- LLM stages: induce session/claude-opus-5.5, judge session/claude-opus-5.5
- Cost: £0.00
- Substitutions: []
