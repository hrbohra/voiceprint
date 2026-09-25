# Voice of AmazonHelp

Corpus: twcs · 2000 target turns · 106365 reference turns · tier `local-first`

## Rules

| # | Rule | When | Strength | Evidence |
|---|---|---|---|---|
| R01 | End every reply with a space, a caret and the agent's initials, e.g. ' ^JK'. | always | usually | held-out: 100% vs ref 17% |
| R02 | When something has gone wrong, open with an apology for the specific problem ('I'm sorry about the delay', 'Sorry for the trouble', 'My apologies') before anything else. | always | sometimes | held-out: 25% vs ref 4% |
| R03 | Move the issue to a private channel with a link: 'Please share your details / reach us here: <URL> and we'll look into it.' | always | often | held-out: 35% vs ref 8% |
| R06 | Speak as 'we' and promise follow-up ('We'll look into this and get back to you shortly'). | always | sometimes | held-out: 25% vs ref 6% |
| R07 | Skip casual greetings ('Hi', 'Hey'); begin with the apology or the answer. | always | usually | held-out: 92% vs ref 73% |
| R09 | Use formal service formulae: 'Kindly', 'correspondence', 'revert', 'the same', 'Please be assured'. | always | sometimes | held-out: 12% vs ref 0% |
| S01 | Keep the tone sober and concerned rather than upbeat | always | often | target -0.0955 (95% CI -0.118–-0.0687) vs reference 0.268; Hedges g -0.74, Cliff's δ -0.41 |
| S02 | Apologise readily | always | sometimes | target 0.357 (95% CI 0.336–0.386) vs reference 0.133; Hedges g 0.59, Cliff's δ 0.22; also measured by emotion_remorse |
| S03 | Skip greetings | always | usually | target 0.0727 (95% CI 0.0594–0.0856) vs reference 0.259; Hedges g -0.47, Cliff's δ -0.18 |
| S04 | Use longer sentences | always | sometimes | target 10.3 (95% CI 10.1–10.5) vs reference 9.5; Hedges g 0.18, Cliff's δ 0.15 |

<details><summary>9 candidate rules failed verification</summary>

- Warn the customer not to post order or account details because the Twitter page is public. ({'target_compliance': 0.042, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.2474, 'judge': 'session/claude-opus-5.5'})
- State that accounts cannot be accessed over Twitter before redirecting. ({'target_compliance': 0.021, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.5, 'judge': 'session/claude-opus-5.5'})
- Put the customer's first name after a comma inside the apology, not as the first word. ({'target_compliance': 0.062, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.1211, 'judge': 'session/claude-opus-5.5'})
- For a late or missing package, ask what the tracking shows, who the carrier is, or whether the promised delivery date has passed. ({'target_compliance': 0.062, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.1211, 'judge': 'session/claude-opus-5.5'})
- Ask the customer to describe the problem without sharing personal or account information. ({'target_compliance': 0.021, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.5, 'judge': 'session/claude-opus-5.5'})
- Acknowledge feelings with 'I understand your concern' or 'I understand your frustration'. ({'target_compliance': 0.062, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.1211, 'judge': 'session/claude-opus-5.5'})
- Close with a courtesy line such as 'Appreciate your patience' or 'Keep us posted!' ({'target_compliance': 0.042, 'reference_compliance': 0.021, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.5, 'judge': 'session/claude-opus-5.5'})
- If a reply runs long, split it across tweets numbered '(1/2)', '(2/2)'. ({'target_compliance': 0.125, 'reference_compliance': 0.042, 'n_target': 48, 'n_reference': 48, 'fisher_p': 0.9705, 'judge': 'session/claude-opus-5.5'})
- Say that feedback has been forwarded internally or to the concerned team when the customer complains. ({'target_compliance': 0.0, 'reference_compliance': 0.0, 'n_target': 48, 'n_reference': 48, 'fisher_p': 1.0, 'judge': 'session/claude-opus-5.5'})

</details>

## Most distinctive measurements

| Feature | Target | Reference | Hedges g | Cliff's δ |
|---|---|---|---|---|
| ends_without_punctuation | 0.975 [0.965, 0.983] | 0.521 | 1.03 | 0.45 |
| sentiment_score | -0.0954 [-0.118, -0.0687] | 0.268 | -0.74 | -0.41 |
| sentiment_negative | 0.277 [0.263, 0.293] | 0.118 | 0.65 | 0.42 |
| sentiment_positive | 0.182 [0.167, 0.194] | 0.386 | -0.63 | -0.41 |
| emotion_sadness_p | 0.0784 [0.0722, 0.0838] | 0.0268 | 0.61 | 0.26 |
| social_apologise | 0.357 [0.336, 0.386] | 0.133 | 0.59 | 0.22 |
| emotion_remorse | 0.344 [0.322, 0.364] | 0.123 | 0.60 | 0.22 |
| semicolons_colons | 0.209 [0.194, 0.224] | 0.0789 | 0.54 | 0.26 |
| emotion_embarrassment_p | 0.00331 [0.0031, 0.00354] | 0.00151 | 0.53 | 0.25 |
| emotion_remorse_p | 0.241 [0.225, 0.257] | 0.0868 | 0.60 | 0.19 |
| pos_propn | 0.0249 [0.0223, 0.0273] | 0.0511 | -0.44 | -0.26 |
| emotion_grief_p | 0.00275 [0.00256, 0.00292] | 0.00138 | 0.53 | 0.17 |
| named_entities_rate | 0.0186 [0.017, 0.0204] | 0.0403 | -0.43 | -0.24 |
| act_confidence | 0.711 [0.703, 0.719] | 0.78 | -0.41 | -0.25 |
| emotion_disappointment_p | 0.0193 [0.0163, 0.0224] | 0.0086 | 0.22 | 0.38 |
| emotion_excitement_p | 0.00509 [0.004, 0.00631] | 0.0152 | -0.21 | -0.38 |
| flesch_reading_ease | 76 [75.2, 76.9] | 82.2 | -0.40 | -0.23 |
| social_greet | 0.0727 [0.0594, 0.0856] | 0.259 | -0.47 | -0.18 |
| emotion_disapproval_p | 0.0252 [0.021, 0.0302] | 0.0147 | 0.17 | 0.36 |
| lexical_density | 0.401 [0.397, 0.406] | 0.431 | -0.34 | -0.20 |
| emotion_realization_p | 0.0104 [0.0101, 0.0108] | 0.00852 | 0.21 | 0.29 |
| flesch_kincaid_grade | 5.41 [5.26, 5.54] | 4.58 | 0.30 | 0.19 |
| parentheses | 0.0887 [0.0707, 0.112] | 0.0121 | 0.45 | 0.08 |
| past_tense_share | 0.273 [0.255, 0.288] | 0.178 | 0.31 | 0.17 |
| emotion_joy_p | 0.0194 [0.0157, 0.0246] | 0.0482 | -0.21 | -0.24 |

## Words and phrases the target over-uses

`here name` (20.24), `order` (18.75), `i'm` (18.3), `i'm sorry` (17.34), `sorry` (16.71), `details` (16.2), `the` (13.56), `2` (13.38), `sorry for` (13.03), `your order` (12.58), `your details` (12.11), `here` (11.82), `provide` (11.77), `for the` (11.73), `us here` (11.56), `delivery` (11.5), `to you` (11.4), `kindly` (11.18), `please` (10.47), `received` (10.3), `provided` (10.26), `1 2` (10.21), `personal` (9.95), `provide your` (9.9), `page` (9.66), `twitter` (9.58), `date` (9.56), `reach` (9.44), `chat` (9.36), `support team` (9.35)

## …and under-uses

`dm` (-13.4), `a` (-12.17), `send` (-11.02), `there` (-10.86), `can` (-10.31), `hi` (-10.1), `us a` (-10.04), `send us` (-9.57), `address` (-9.22), `help` (-8.59), `hey` (-7.93), `with your` (-7.87), `up` (-7.86), `we can` (-7.85), `let's` (-7.32), `happy` (-7.31), `and` (-7.27), `to help` (-7.12), `me` (-7.03), `happy to` (-6.95)

## Dialogue habits

- Starts 100% of conversations, ends 34%.
- Sends multiple messages in a row 10% of the time (mean burst 1.12).
- Top openers: `i'm sorry` ×3380, `oh no` ×682, `thanks for` ×535, `sorry to` ×528, `please don't` ×525, `i'm so` ×513
- Top closers: `<URL> gg` ×102, `this hn` ×93, `<URL> wm` ×85, `<URL> ag` ×83, `<URL> wt` ×82, `<URL> dw` ×80

## Situations

| Situation | Turns | Keywords |
|---|---|---|
| account or service help request | 862 | amazon, please, help, get, service, account, customer, email |
| order and delivery problem | 762 | order, delivery, delivered, amazon, package, today, day, prime |

## Privacy

- 634617 turns in, 0 quarantined, replacements: {}.
- Exemplars are k-anonymous (k=3); memorisation check: 0 leaks.

## Run

- Models: cuda · spaCy en_core_web_trf-3.8.0
- LLM stages: induce session/claude-opus-5.5, judge session/claude-opus-5.5
- Cost: £0.00
- Substitutions: []
