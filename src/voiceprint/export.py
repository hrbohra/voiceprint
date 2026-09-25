"""Everything a run produces, written to one output directory:

  voice.json        machine-readable profile: statistics, contrast, keyness, rules with evidence,
                    situations, dialogue habits, scorer reference, manifest.
  VOICE.md          the same for humans.
  prompt_pack/      system prompts at three lengths (compact ~150 tokens, standard ~600, full), plus
                    few-shot exemplars. For prompting any LLM today, with no training.
  exemplars.jsonl   cleared exemplar turns with situation and context, for retrieval-time few-shot.
  sft.jsonl         supervised fine-tuning pairs (context -> target reply) in chat format.
  dpo.jsonl         preference pairs: target reply (chosen) vs a reference reply to a similar
                    situation (rejected). Only when a reference exists.

Every text written here passes the memorisation check first; a leak fails the run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .rules import Rule

STRENGTH_WORD = {"always": "Always", "usually": "Usually", "often": "Often", "sometimes": "Sometimes", "rarely": "Rarely", "never": "Never"}


def _rule_line(r: Rule, detail: bool) -> str:
    scope = "" if r.situation == "always" else f" (when: {r.situation})"
    s = r.statement.rstrip(".")
    lead = "" if s.lower().startswith(STRENGTH_WORD[r.strength].lower()) or r.strength in ("usually",) else f"{STRENGTH_WORD[r.strength]}: "
    line = f"- {lead}{s}{scope}."
    if detail and r.example:
        line += f"\n  e.g. \"{r.example}\""
        if r.counter_example:
            line += f"  not \"{r.counter_example}\""
    return line


def prompt_pack(name: str, rules: list[Rule], dialogue: dict, exemplars: list[dict]) -> dict[str, str]:
    kept = [r for r in rules if r.kept]
    top = kept[:8]
    openers = ", ".join(f'"{o}"' for o, _ in dialogue.get("top_openers", [])[:4] if o != "∅")
    compact = f"Write as {name}. " + " ".join(f"{r.statement.rstrip('.')}." for r in top)
    standard = "\n".join([
        f"You write in the voice of {name}. Follow these voice rules; they were measured from real messages and verified on held-out ones.",
        "", *(_rule_line(r, False) for r in kept[:20]),
        "", f"Typical openings: {openers}." if openers else "",
        "Keep the facts of the reply accurate; the rules govern how you say it, not what is true.",
    ]).strip()
    shots = "\n\n".join(f"[Replying to] {e['context']}\n[{name}] {e['text']}" if e.get("context") else f"[{name}] {e['text']}" for e in exemplars[:12])
    full = "\n".join([standard, "", "Rules with examples:", *(_rule_line(r, True) for r in kept), "", "Examples of the voice (anonymised):", shots]).strip()
    return {"compact.txt": compact, "standard.txt": standard, "full.txt": full}


def voice_md(name: str, data: dict) -> str:
    L = [f"# Voice of {name}", "", f"Corpus: {data['corpus']['name']} · {data['corpus']['target_turns']} target turns · "
         f"{data['corpus']['reference_turns']} reference turns · tier `{data['manifest']['tier']}`", ""]
    L += ["## Rules", "", "| # | Rule | When | Strength | Evidence |", "|---|---|---|---|---|"]
    for r in data["rules"]:
        if not r["kept"]:
            continue
        v = r["verification"]
        if "target_compliance" in v:
            ev = f"held-out: {v['target_compliance']:.0%} vs ref {v['reference_compliance']:.0%}" if v.get("reference_compliance") is not None else f"held-out: {v['target_compliance']:.0%}"
        else:
            ev = r["rationale"]
        L.append(f"| {r['id']} | {r['statement']} | {r['situation']} | {r['strength']} | {ev} |")
    dropped = [r for r in data["rules"] if not r["kept"]]
    if dropped:
        L += ["", f"<details><summary>{len(dropped)} candidate rules failed verification</summary>", ""]
        L += [f"- {r['statement']} ({r['verification']})" for r in dropped]
        L += ["", "</details>"]
    L += ["", "## Most distinctive measurements", "", "| Feature | Target | Reference | Hedges g | Cliff's δ |", "|---|---|---|---|---|"]
    for f, r in list(data["contrast"].items())[:25]:
        L.append(f"| {f} | {r['target_mean']:.3g} [{r['target_ci_lo']:.3g}, {r['target_ci_hi']:.3g}] | {r['reference_mean']:.3g} | {r['hedges_g']:.2f} | {r['cliffs_delta']:.2f} |")
    k = data.get("keyness", {})
    if k.get("over"):
        L += ["", "## Words and phrases the target over-uses", "", ", ".join(f"`{x['term']}` ({x['z']})" for x in k["over"][:30])]
    if k.get("under"):
        L += ["", "## …and under-uses", "", ", ".join(f"`{x['term']}` ({x['z']})" for x in k["under"][:20])]
    d = data.get("dialogue", {})
    if d:
        L += ["", "## Dialogue habits", "",
              f"- Starts {d['starts_conversation_rate']:.0%} of conversations, ends {d['ends_conversation_rate']:.0%}.",
              f"- Sends multiple messages in a row {d['multi_message_turn_rate']:.0%} of the time (mean burst {d['mean_burst']:.2f}).",
              f"- Top openers: {', '.join(f'`{o}` ×{c}' for o, c in d['top_openers'][:6])}",
              f"- Top closers: {', '.join(f'`{o}` ×{c}' for o, c in d['top_closers'][:6])}"]
    if data.get("situations"):
        L += ["", "## Situations", "", "| Situation | Turns | Keywords |", "|---|---|---|"]
        L += [f"| {s['name'] or s['id']} | {s['size']} | {', '.join(s['keywords'])} |" for s in data["situations"]]
    p = data.get("privacy", {})
    if p:
        L += ["", "## Privacy", "", f"- {p.get('turns_in', 0)} turns in, {p.get('quarantined', 0)} quarantined, replacements: {p.get('replaced', {})}.",
              f"- Exemplars are k-anonymous (k={data['manifest']['privacy']['k_anonymity']}); memorisation check: {data['manifest'].get('memorisation_leaks', 0)} leaks."]
    m = data["manifest"]
    L += ["", "## Run", "", f"- Models: {m.get('models', {}).get('device')} · spaCy {m.get('models', {}).get('spacy')}",
          f"- LLM stages: {m.get('llm', 'none')}", f"- Cost: £{m.get('cost', {}).get('gbp', 0):.2f}", f"- Substitutions: {m.get('substitutions', [])}"]
    return "\n".join(L) + "\n"


def write_all(out: Path, name: str, data: dict, exemplars: list[dict], sft: list[dict], dpo: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_pack").mkdir(exist_ok=True)
    rules = [Rule.model_validate(r) for r in data["rules"]]
    for fn, txt in prompt_pack(name, rules, data.get("dialogue", {}), exemplars).items():
        (out / "prompt_pack" / fn).write_text(txt + "\n", encoding="utf-8", newline="\n")
    (out / "voice.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n")
    (out / "VOICE.md").write_text(voice_md(name, data), encoding="utf-8", newline="\n")
    for fn, rows in (("exemplars.jsonl", exemplars), ("sft.jsonl", sft), ("dpo.jsonl", dpo)):
        with (out / fn).open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
