# Voice of target_host

Corpus: planted · 120 target turns · 231 reference turns · tier `local-first`

## Rules

| # | Rule | When | Strength | Evidence |
|---|---|---|---|---|
| R01 | Open by addressing the guest by first name followed by an exclamation mark. | always | usually | held-out: 92% vs ref 0% |
| R02 | Write in lowercase, including sentence starts and 'i', capitalising only the guest's name. | always | usually | held-out: 76% vs ref 0% |
| R03 | End on a single 🙂 after the last word, with no full stop before it. | always | often | held-out: 56% vs ref 0% |
| R04 | Use contractions wherever possible (it's, there's, you're, i'll, that's). | always | usually | held-out: 100% vs ref 20% |
| R05 | Soften the answer with 'i think' at the start of the main clause, even for plain facts. | always | often | held-out: 64% vs ref 0% |
| R06 | After the answer, offer more help with the question 'anything else you need?'. | always | sometimes | held-out: 56% vs ref 0% |
| R07 | Grant permission with 'you're free to' or 'you're welcome to'. | always | sometimes | held-out: 20% vs ref 0% |
| R09 | Keep the tone warm and upbeat rather than neutral. | always | usually | held-out: 92% vs ref 37% |
| S01 | Write in all lowercase | always | usually | target 0.853 (95% CI 0.788–0.914) vs reference 0; Hedges g 4.10, Cliff's δ 0.85; also measured by sentences |
| S02 | Open by addressing the reader by name | always | usually | target 0.832 (95% CI 0.732–0.907) vs reference 0; Hedges g 3.79, Cliff's δ 0.83 |
| S03 | Put emoji at the end of the message | always | usually | target 0.779 (95% CI 0.691–0.861) vs reference 0; Hedges g 3.20, Cliff's δ 0.78; also measured by has_emoji, emoji_count |
| S04 | Use contractions (I'm, we'll, don't) | always | usually | target 0.0629 (95% CI 0.0555–0.07) vs reference 0; Hedges g 2.94, Cliff's δ 0.82 |
| S05 | Soften claims with hedges (maybe, I think) | always | usually | target 0.0345 (95% CI 0.0278–0.0418) vs reference 0; Hedges g 1.82, Cliff's δ 0.55 |
| S06 | Use exclamation marks freely | always | usually | target 0.753 (95% CI 0.66–0.842) vs reference 0.184; Hedges g 1.45, Cliff's δ 0.62 |
| S07 | Write longer messages than typical | always | usually | target 15.6 (95% CI 15.1–16.2) vs reference 12.4; Hedges g 1.41, Cliff's δ 0.65 |
| S08 | End messages with a question | always | often | target 0.463 (95% CI 0.372–0.566) vs reference 0; Hedges g 1.58, Cliff's δ 0.46; also measured by emotion_curiosity, questions_per_sentence, act_ask |
| S09 | Let feeling show rather than staying neutral | always | usually | target 0.179 (95% CI 0.101–0.259) vs reference 0.762; Hedges g -1.41, Cliff's δ -0.58 |
| S10 | Write informally (verb- and pronoun-heavy) | always | usually | target 52.9 (95% CI 50.4–55.4) vs reference 63.3; Hedges g -0.99, Cliff's δ -0.57 |
| S11 | Answer a question with a question back | always | sometimes | target 0.347 (95% CI 0.259–0.45) vs reference 0; Hedges g 1.24, Cliff's δ 0.35 |
| S12 | Reply at greater length than the message received | always | often | target 1.72 (95% CI 1.66–1.8) vs reference 1.39; Hedges g 0.95, Cliff's δ 0.47 |
| S13 | Keep the tone warm and positive | always | often | target 0.789 (95% CI 0.744–0.828) vs reference 0.522; Hedges g 0.90, Cliff's δ 0.46 |
| S14 | Offer further help | always | sometimes | target 0.411 (95% CI 0.303–0.51) vs reference 0.0919; Hedges g 0.85, Cliff's δ 0.32 |
| S15 | Keep your own style regardless of the other person | always | often | target 0.5 (95% CI 0.472–0.535) vs reference 0.598; Hedges g -0.73, Cliff's δ -0.35 |
| S16 | Use longer sentences | always | often | target 14.1 (95% CI 13.3–14.9) vs reference 12.4; Hedges g 0.59, Cliff's δ 0.38 |
| S17 | Respond with more than a bare answer | always | often | target 0.4 (95% CI 0.315–0.5) vs reference 0.659; Hedges g -0.54, Cliff's δ -0.26; also measured by act_acknowledge |
| S18 | Rarely express approval | always | often | target 0.368 (95% CI 0.293–0.457) vs reference 0.589; Hedges g -0.45, Cliff's δ -0.22 |

<details><summary>3 candidate rules failed verification</summary>

- Answer in one or two short clauses joined by 'and' or a comma, with no elaboration. ({'target_compliance': 1.0, 'reference_compliance': 1.0, 'n_target': 25, 'n_reference': 46, 'fisher_p': 1.0, 'judge': 'session/claude-opus-5.5'})
- Greet the other person ({'statistical': True, 'hedges_g': 0.517, 'cliffs_delta': 0.084, 'heldout_target_mean': 0.0, 'heldout_reference_mean': 0.0})
- Use the passive voice ({'statistical': True, 'hedges_g': 0.437, 'cliffs_delta': 0.074, 'heldout_target_mean': 0.0, 'heldout_reference_mean': 0.0})

</details>

## Most distinctive measurements

| Feature | Target | Reference | Hedges g | Cliff's δ |
|---|---|---|---|---|
| all_lowercase | 0.853 [0.788, 0.914] | 0 | 4.10 | 0.85 |
| pos_propn | 0.0482 [0.0433, 0.053] | 0 | 3.56 | 0.83 |
| opens_with_name | 0.832 [0.732, 0.907] | 0 | 3.79 | 0.83 |
| emoji_end | 0.779 [0.691, 0.861] | 0 | 3.20 | 0.78 |
| has_emoji | 0.779 [0.696, 0.864] | 0 | 3.20 | 0.78 |
| emoji_count | 0.779 [0.696, 0.868] | 0 | 3.20 | 0.78 |
| contraction_rate | 0.0629 [0.0555, 0.07] | 0 | 2.94 | 0.82 |
| emotion_surprise_p | 0.00393 [0.00356, 0.00442] | 0.00076 | 2.60 | 0.94 |
| emotion_love_p | 0.00694 [0.00611, 0.00793] | 0.00214 | 2.10 | 0.91 |
| chars | 81.6 [79, 84.2] | 59.4 | 2.08 | 0.81 |
| emotion_desire_p | 0.0124 [0.0108, 0.014] | 0.00183 | 1.68 | 0.90 |
| emotion_nervousness_p | 0.00087 [0.00078, 0.00097] | 0.00037 | 1.77 | 0.75 |
| emotion_curiosity_p | 0.238 [0.18, 0.29] | 0.00186 | 1.52 | 0.86 |
| hedge_rate | 0.0345 [0.0278, 0.0418] | 0 | 1.82 | 0.55 |
| emotion_amusement_p | 0.00529 [0.00443, 0.00626] | 0.00126 | 1.36 | 0.77 |
| emotion_confusion_p | 0.0196 [0.0159, 0.0235] | 0.00272 | 1.53 | 0.59 |
| exclamations_per_sentence | 0.753 [0.66, 0.842] | 0.184 | 1.45 | 0.62 |
| emotion_excitement_p | 0.0506 [0.0374, 0.0662] | 0.00597 | 1.16 | 0.83 |
| words | 15.6 [15.1, 16.2] | 12.4 | 1.41 | 0.65 |
| emotion_optimism_p | 0.0534 [0.0433, 0.0656] | 0.00966 | 1.16 | 0.79 |
| ends_with_question | 0.463 [0.372, 0.566] | 0 | 1.58 | 0.46 |
| emotion_neutral | 0.179 [0.101, 0.259] | 0.762 | -1.41 | -0.58 |
| tree_depth_mean | 2.79 [2.63, 2.97] | 3.73 | -1.28 | -0.68 |
| emotion_neutral_p | 0.205 [0.173, 0.241] | 0.559 | -1.27 | -0.60 |
| ends_without_punctuation | 0.432 [0.333, 0.542] | 0 | 1.49 | 0.43 |

## Words and phrases the target over-uses

`think` (5.75), `i think` (5.75), `anything` (5.28), `else` (5.28), `need` (5.28), `anything else` (5.28), `else you` (5.28), `you need` (5.28), `it's` (3.97), `i` (3.87), `there's` (3.81), `you're` (3.55), `and you're` (3.55), `i'll` (2.97), `there's a` (2.97), `tom` (2.86), `priya` (2.75), `and it's` (2.63), `it's open` (2.63), `ravi` (2.63), `problem i'll` (2.63), `i'll be` (2.63), `think it's` (2.51), `you're welcome` (2.51), `you're free` (2.51), `bike there's` (2.38), `there's space` (2.38), `aisha` (2.38), `maya` (2.24), `ravi i` (2.24)

## …and under-uses

`is` (-5.98), `there` (-4.81), `are` (-4.32), `there is` (-4.12), `is a` (-4.12), `it is` (-3.88), `you are` (-3.42), `and you` (-3.42), `i will` (-3.41), `will` (-3.41), `the` (-3.27), `leave` (-3.01), `it` (-2.89), `in the` (-2.75), `in` (-2.75), `will leave` (-2.49), `fine i` (-2.49), `is fine` (-2.49), `that is` (-2.49), `that` (-2.49)

## Dialogue habits

- Starts 0% of conversations, ends 100%.
- Sends multiple messages in a row 0% of the time (mean burst 1.00).
- Top openers: `ravi i` ×5, `priya i` ×4, `maya i` ×3, `jon i` ×2, `tom i` ×2, `leo i` ×2
- Top closers: `you need` ×15, `host you` ×5, `open early` ×3, `minutes away` ×3, `any time` ×3, `the hallway` ×3

## Situations

| Situation | Turns | Keywords |
|---|---|---|
| late arrival or delay | 14 | sorry, train, delayed, i'll, hello, check, bit, late |
| breakfast recommendation request | 12 | recommend, somewhere, breakfast, nearby |
| bringing items or leaving bags | 12 | mind, bring, bike, leave, bags, check-out |
| availability check for dates | 11 | room, free |
| thanks after a stay | 10 | much, lovely |
| washing machine question | 10 | washing, machine, use |
| neighbourhood walkability question | 8 | neighbourhood, easy, walk |
| noise at night concern | 7 | quiet, night, i'm, light, sleeper |
| spare towel request | 6 | spare, towel |
| wifi and remote work question | 5 | what's, wifi, like, need, work, flat |

## Privacy

- 702 turns in, 0 quarantined, replacements: {}.
- Exemplars are k-anonymous (k=3); memorisation check: 0 leaks.

## Run

- Models: cuda · spaCy en_core_web_trf-3.8.0
- LLM stages: induce session/claude-opus-5.5, judge session/claude-opus-5.5
- Cost: £0.00
- Substitutions: []
