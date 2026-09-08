#!/usr/bin/env python3
"""Report what a backend can actually do, before you build on it.

The one question that decides the architecture is whether the runtime will
score a string you supply. Stage 2 needs only generation with logprobs, which
most runtimes have. Stage 1 (PMI) needs teacher forcing, which several do not.
Guessing here wastes a day; this takes seconds.

  python check_backend.py --backend ollama   --model gemma3:27b
  python check_backend.py --backend llamacpp --model gemma3:27b
  python check_backend.py --backend transformers --model google/gemma-3-27b-it
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import (BackendError, OllamaBackend, discover_ollama_host,  # noqa
                      docker_cp_command, docker_ollama_containers,
                      docker_ollama_store, ollama_models_root,
                      resolve_ollama_gguf)
from prompts import LABELS  # noqa: E402


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def check_ollama(args):
    host = args.host or "http://localhost:11434"
    be = OllamaBackend(args.model, host=host, top_logprobs=args.top_logprobs)
    r = be.probe(LABELS)

    print(f"host          : {r['host']}")
    if not r["reachable"]:
        print(f"reachable     : NO")
        print(f"                {r['error']}")
        # In Docker, "localhost" may be the wrong side of the network.
        found = discover_ollama_host()
        if found and found != be.host:
            print(f"\n  Found one at {found} instead. Re-run with:")
            print(f"      --host {found}")
            print( "  (localhost points at the calling container when the caller")
            print( "   is itself containerised; host.docker.internal on Docker")
            print( "   Desktop and 172.17.0.1 on Linux reach the host.)")
            return 1
        containers = docker_ollama_containers()
        if containers:
            print(f"\n  Ollama container(s) running: {', '.join(containers)}")
            print( "  but nothing answers on the tried hosts. The port is likely")
            print( "  not published. Start it with:")
            print( "      docker run -d -p 11434:11434 -v ~/.ollama:/root/.ollama \\")
            print( "                 --name ollama ollama/ollama")
        else:
            print("\n  No Ollama container found either.  ollama serve")
        return 1
    print(f"reachable     : yes (ollama {r['version']})")

    present = r.get("model_present")
    print(f"model         : {r['model']}"
          f"{'' if present is None else ('  [installed]' if present else '  [NOT FOUND]')}")
    if present is False and r.get("available_models"):
        print(f"                installed models: {', '.join(r['available_models'][:8])}")
        print(f"                pull it with:  ollama pull {r['model']}")

    print()
    print("=== stage 2 (A-E labels) ===")
    if r["logprobs"]:
        print(f"  logprobs    : yes")
        print(f"  labels seen : {', '.join(r['labels_found'])}")
        if r["labels_missing"]:
            print(f"  labels MISSING from top-{args.top_logprobs}: "
                  f"{', '.join(r['labels_missing'])}")
            print( "                raise --top-logprobs, or expect those to be")
            print( "                floored during scoring")
        print("  verdict     : stage 2 works")
    else:
        print(f"  logprobs    : NO")
        if r["error"]:
            print(f"  {r['error']}")
        print("  verdict     : stage 2 unavailable. Needs Ollama v0.12.11+ and a")
        print("                local model; Ollama Cloud has been reported to")
        print("                return null logprobs.")

    print()
    print("=== stage 1 (PMI over supplied symbols) ===")
    print("  verdict     : NOT AVAILABLE through Ollama.")
    print("                Its logprobs cover generated tokens only. There is no")
    print("                echo option and no scoring endpoint, so the probability")
    print("                of a symbol you supply cannot be read out.")
    print()
    print("  This matters: without stage 1 nothing subtracts corpus frequency,")
    print("  so famous genes drift to the top for every disease. Stage 2 alone")
    print("  still never generates a symbol, but check the frequency-only")
    print("  control in evaluate.py before trusting the ranking.")

    print()
    print("=== same weights, with scoring ===")
    containers = docker_ollama_containers()
    if containers:
        print(f"  docker      : Ollama is containerised ({', '.join(containers)})")
        store = docker_ollama_store()
        print(f"  model store : "
              f"{store or 'not readable from here (named volume or Docker Desktop VM)'}")
    try:
        blob = resolve_ollama_gguf(args.model)
        size = os.path.getsize(blob)
        print(f"  GGUF found  : {blob}  ({human(size)})")
        print(f"  Ollama already has these weights on disk. llama.cpp can score")
        print(f"  them, so the full pipeline runs with no second download:")
        print(f"      pip install llama-cpp-python")
        print(f"      python rank.py --backend llamacpp --model {args.model} ...")
    except BackendError as e:
        print(f"  GGUF lookup : not resolved")
        for line in str(e).split("\n"):
            print(f"    {line}")
        if containers and "Docker" not in str(e):
            # the resolver already explains the Docker case when it detects it
            cmd = docker_cp_command(args.model)
            print()
            print("  Two ways to make the weights readable:")
            print("    1. bind-mount the store instead of using a named volume,")
            print("       then restart the container (no file is duplicated):")
            print("           docker run -d -p 11434:11434 \\")
            print("                      -v ~/.ollama:/root/.ollama \\")
            print("                      --name ollama ollama/ollama")
            if cmd:
                print("    2. copy the GGUF out (duplicates it - a 27B Q4 is ~17GB):")
                print(f"           {cmd}")
                print("       then:  --backend llamacpp --model ./<file>.gguf")
    return 0


def check_llamacpp(args):
    print(f"ollama store  : {ollama_models_root()}")
    dstore = docker_ollama_store()
    if dstore:
        print(f"docker store  : {dstore}")
    elif docker_ollama_containers():
        print("docker store  : Ollama is in Docker but its store is not readable "
              "from here")
    path = args.model
    if not os.path.exists(path):
        try:
            path = resolve_ollama_gguf(args.model)
        except BackendError as e:
            print(f"GGUF          : NOT FOUND\n\n{e}")
            return 1
    print(f"GGUF          : {path}  ({human(os.path.getsize(path))})")

    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        print("llama-cpp-python : NOT INSTALLED")
        print("  pip install llama-cpp-python")
        print("\nBoth stages would be available once it is installed.")
        return 1
    print(f"llama-cpp-python : {llama_cpp.__version__}")

    if args.no_load:
        print("\n--no-load given; skipping the actual model load.")
        return 0

    from backends import LlamaCppBackend
    print("\nloading (this reads the whole file)...")
    be = LlamaCppBackend(model_path=path, n_ctx=args.n_ctx,
                         n_gpu_layers=args.n_gpu_layers)

    print("\n=== stage 1 (PMI over supplied symbols) ===")
    scores = be.score_continuations(
        "Gene most strongly associated with Cystic fibrosis: ",
        ["CFTR", "GPR52", "GPR56"])
    for g, s in zip(["CFTR", "GPR52", "GPR56"], scores):
        print(f"  logP({g:<6}) = {s:.3f}")
    ok1 = len(set(round(float(x), 6) for x in scores)) > 1
    print(f"  verdict     : {'stage 1 works' if ok1 else 'SUSPECT - identical scores'}")
    if not ok1:
        print("                Identical scores mean the supplied text is not")
        print("                reaching the model. Do not build on this.")

    print("\n=== stage 2 (A-E labels) ===")
    got = be.label_logprobs(
        "Disease: Cystic fibrosis\nOptions:\nA. HBB\nB. CFTR\n"
        "C. None of the above\nAnswer:", LABELS)
    for lab in LABELS:
        v = got[lab]
        print(f"  {lab}: {'missing' if v is None else f'{v:.3f}'}")
    print(f"  verdict     : "
          f"{'stage 2 works' if any(v is not None for v in got.values()) else 'no logprobs'}")
    return 0


def check_transformers(args):
    for mod in ("torch", "transformers"):
        try:
            m = __import__(mod)
            print(f"{mod:<14}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<14}: NOT INSTALLED  ->  pip install torch transformers")
            return 1
    print("\nBoth stages available. This is the reference path.")
    print("Run check_tokenizer.py next to see how the symbols tokenize.")
    return 0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--backend", required=True,
                    choices=["ollama", "llamacpp", "transformers"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--no-load", action="store_true",
                    help="llamacpp: report without loading the weights")
    args = ap.parse_args()

    print(f"backend       : {args.backend}\n")
    fn = {"ollama": check_ollama, "llamacpp": check_llamacpp,
          "transformers": check_transformers}[args.backend]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
