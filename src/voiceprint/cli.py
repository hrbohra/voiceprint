"""Command line.

    voiceprint doctor                         check GPU, models, providers (no keys printed)
    voiceprint inspect CORPUS                 formats, speakers, turn counts
    voiceprint extract CORPUS -s SPEAKER -o out/   the full run
    voiceprint score out/voice.json "text"    score a message against a voice
    voiceprint eval planted|attribution|curve ...  the evaluation suite (see eval/)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; outputs contain emoji and arrows
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Extract, verify and export the voice of a speaker from a text corpus.")
con = Console()


@app.command()
def doctor() -> None:
    """Environment self-test: device, spaCy pipeline, LLM providers available (never prints keys)."""
    from . import models
    from .llm.providers import available

    models.configure(True)
    t = Table("check", "result")
    t.add_row("device", models.device())
    try:
        t.add_row("spaCy", models.spacy_name())
    except Exception as e:  # noqa: BLE001
        t.add_row("spaCy", f"[red]{e}[/red]")
    for p, ok in available().items():
        t.add_row(f"provider {p}", "[green]configured[/green]" if ok else "[dim]no credentials[/dim]")
    con.print(t)


@app.command()
def inspect(corpus: Path, fmt: Optional[str] = typer.Option(None, help="jsonl|csv|prose|whatsapp|meta|twitter")) -> None:
    """Show what a corpus loads as."""
    from .ingest import load

    c = load(corpus, fmt)
    con.print(f"[bold]{c.name}[/bold]: {len(c)} turns, {len(c.docs())} conversations")
    t = Table("speaker", "turns")
    for s, n in list(c.speakers().items())[:25]:
        t.add_row(s, str(n))
    con.print(t)


@app.command()
def extract(
    corpus: Path,
    speaker: list[str] = typer.Option([], "--speaker", "-s", help="Target speaker (repeatable). Default: the most frequent."),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    fmt: Optional[str] = typer.Option(None, help="Force a loader."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Statistical rules only; nothing leaves the machine."),
    public: bool = typer.Option(False, "--public", help="Corpus is public (skips anonymisation, allows any provider)."),
    max_turns: Optional[int] = typer.Option(None, help="Use only the first N turns (for quick runs)."),
    cpu: bool = typer.Option(False, help="Force CPU."),
) -> None:
    """Run the full extraction and write voice.json, VOICE.md, the prompt pack and training files."""
    from . import pipeline
    from .config import Config
    from .ingest import load
    from .schema import Corpus

    logging.basicConfig(level=logging.WARNING)
    cfg = Config.load(config)
    if cpu:
        cfg.use_gpu = False
    c = load(corpus, fmt)
    if max_turns:
        c = Corpus(c.turns[:max_turns], c.name)
    res = pipeline.run(c, cfg, targets=speaker or None, use_llm=not no_llm, public=public, progress=con.print)
    pipeline.write(res, out)
    kept = [r for r in res.data["rules"] if r["kept"]]
    con.print(f"\n[bold green]done[/bold green]: {len(kept)} verified rules → {out / 'VOICE.md'}")
    if res.data.get("scorer_eval"):
        con.print(f"scorer held-out AUC: {json.dumps(res.data['scorer_eval'])}")


@app.command()
def score(voice: Path, text: list[str]) -> None:
    """Score one or more messages against an extracted voice (0..1, surface scorer)."""
    from . import scorer

    data = json.loads(voice.read_text(encoding="utf-8"))
    p = scorer.score_surface(data["scorer"], text)
    for t, s in zip(text, p):
        con.print(f"{s:.3f}  {t}")


@app.command("ts-lib")
def ts_lib(out: Path) -> None:
    """Write the TypeScript scorer as a library (lexicons built in, model passed at runtime)."""
    from . import scorer

    out.write_text(scorer.to_typescript(None), encoding="utf-8", newline="\n")
    con.print(f"wrote {out}")


eval_app = typer.Typer(help="Evaluation suite.")
app.add_typer(eval_app, name="eval")


@eval_app.command("planted")
def eval_planted(out: Path = typer.Option(Path("results/planted"), "--out", "-o"), n_docs: int = 120, no_llm: bool = False) -> None:
    """Synthetic corpus with known, planted voice rules: does the extractor recover them?"""
    from .eval import planted

    res = planted.run(out, n_docs=n_docs, use_llm=not no_llm, progress=con.print)
    con.print_json(json.dumps(res["summary"]))


@eval_app.command("attribution")
def eval_attribution(corpus: Path, out: Path = typer.Option(Path("results/attribution"), "--out", "-o"), fmt: Optional[str] = None,
                     speakers: int = 10, min_turns: int = 150, max_turns: int = 400, public: bool = True) -> None:
    """Can the extracted voices tell speakers apart on held-out turns? (authorship attribution)"""
    from .eval import attribution
    from .ingest import load

    res = attribution.run(load(corpus, fmt), out, n_speakers=speakers, min_turns=min_turns, max_turns=max_turns, public=public, progress=con.print)
    con.print_json(json.dumps(res["summary"]))


@eval_app.command("curve")
def eval_curve(corpus: Path, speaker: str, out: Path = typer.Option(Path("results/curve"), "--out", "-o"), fmt: Optional[str] = None,
               sizes: str = "25,50,100,200,400,800", public: bool = True) -> None:
    """Data-efficiency: how the profile and scorer stabilise as the corpus grows."""
    from .eval import curve
    from .ingest import load

    res = curve.run(load(corpus, fmt), speaker, out, [int(s) for s in sizes.split(",")], public=public, progress=con.print)
    con.print_json(json.dumps(res["summary"]))


@eval_app.command("fidelity")
def eval_fidelity(voice: Path, out: Path = typer.Option(Path("results/fidelity"), "--out", "-o"), n: int = 40, public: bool = True) -> None:
    """Generate replies with and without the prompt pack; measure voice score, judge preference and
    content preservation (NLI)."""
    from .eval import fidelity

    res = fidelity.run(voice, out, n=n, public=public, progress=con.print)
    con.print_json(json.dumps(res["summary"]))


if __name__ == "__main__":
    app()
