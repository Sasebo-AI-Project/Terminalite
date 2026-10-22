"""Turn a natural-language request (English or Japanese) into one bash command, on CPU.

The generator's 8-best from beam search is reranked by a reverse model p(request | command) (noisy
channel). Nothing is executed: the command is only printed.

    tl "show the 5 largest files under logs"
    tl -n 3 "logs フォルダで一番大きいファイルは？"
    tl                      # interactive: one request per line
"""

import os
import time

T0 = time.perf_counter()
# OpenBLAS defaults to every core; for these small matrices the thread overhead outweighs it (4 was fastest).
# Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

from . import __version__  # noqa: E402
from .model import MAX_SRC, GRUCopyModel  # noqa: E402
from .weights import FORWARD, REVERSE, ensure  # noqa: E402

TOKENIZER = Path(__file__).resolve().parent / "bpe16k.json"


def suggest(request: str, tok: Tokenizer, fwd: GRUCopyModel, rev: GRUCopyModel, k: int, beta: float) -> tuple[list, dict]:
    """Candidates best first as (command, score), and the time spent in each stage (ms)."""
    t = time.perf_counter()
    hyps = fwd.beam_nbest(tok.encode(request).ids[:MAX_SRC], k)
    cmds = [tok.decode(ids).strip() for ids, _ in hyps]
    t_beam = time.perf_counter()
    chan = rev.score([e.ids[:MAX_SRC] for e in tok.encode_batch(cmds)], tok.encode(request).ids)
    ranked = sorted(((c, s + beta * r) for c, (_, s), r in zip(cmds, hyps, chan)), key=lambda x: -x[1])
    done = time.perf_counter()
    return ranked, {"beam": (t_beam - t) * 1000, "rerank": (done - t_beam) * 1000}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="tl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", nargs="*", help="the request; omit for an interactive prompt")
    # Chosen on the development set for this generator / reverse model pair.
    ap.add_argument("--beta", type=float, default=4.0, help="weight of the reverse model's score")
    ap.add_argument("--k", type=int, default=8, help="beam width")
    ap.add_argument("-n", type=int, default=1, help="number of candidates to print")
    ap.add_argument("--timing", action="store_true", help="print startup and per-stage times to stderr")
    ap.add_argument("--version", action="version", version=f"terminalite {__version__}")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    fwd, rev = GRUCopyModel(ensure(FORWARD)), GRUCopyModel(ensure(REVERSE))
    if args.timing:
        print(f"[startup {(time.perf_counter() - T0) * 1000:.0f} ms]", file=sys.stderr)

    def answer(request: str) -> None:
        ranked, ms = suggest(request, tok, fwd, rev, args.k, args.beta)
        if args.n == 1:
            print(ranked[0][0])
        else:
            for cmd, score in ranked[:args.n]:
                print(f"{score:8.3f}  {cmd}")
        if args.timing:
            print(f"[beam {ms['beam']:.0f} ms, rerank {ms['rerank']:.0f} ms]", file=sys.stderr)
        sys.stdout.flush()

    if args.request:
        answer(" ".join(args.request))
        return
    interactive = sys.stdin.isatty()
    if interactive:
        print("Type a request (English or Japanese); Ctrl+C to quit.", file=sys.stderr)
    while True:
        try:
            line = input("> " if interactive else "")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip():
            answer(line.strip())


if __name__ == "__main__":
    main()
