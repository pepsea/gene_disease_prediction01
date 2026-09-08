#!/usr/bin/env python3
"""Backend tests, including a stand-in Ollama server on a real socket.

The Ollama client is exercised over actual HTTP - request shape, response
parsing, both documented top_logprobs encodings, missing labels, and the
failure paths - so the only thing left untested against a real server is
whether Ollama's wire format matches its own documentation.

    python tests/test_backends.py
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import backends as B  # noqa: E402
from backends import (BackendError, OllamaBackend, ScoringUnsupported,  # noqa
                      discover_ollama_host, docker_cp_command,
                      docker_ollama_containers, docker_ollama_store,
                      match_labels, resolve_ollama_gguf)
from prompts import LABELS  # noqa: E402
from rank import GeneRanker  # noqa: E402

RECEIVED = []          # request payloads the fake server saw
MODE = {"top": "list", "logprobs": True, "labels": set("ABCDE")}


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._json({"version": "0.12.11"})
        if self.path == "/api/tags":
            return self._json({"models": [{"name": "gemma3:27b"}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode())
        RECEIVED.append(payload)
        if self.path != "/api/generate":
            return self._json({"error": "not found"}, 404)
        if not MODE["logprobs"]:
            return self._json({"response": "B", "done": True})

        # CFTR is the "right" answer; give its letter the best logprob.
        best = "A"
        for line in payload["prompt"].rsplit("Options:", 1)[-1].split("\n"):
            line = line.strip()
            if len(line) > 3 and line[1] == "." and "CFTR" in line:
                best = line[0]
        scores = {f" {lab}": (-0.1 if lab == best else -4.0)
                  for lab in sorted(MODE["labels"])}
        top = ([{"token": t, "logprob": v} for t, v in scores.items()]
               if MODE["top"] == "list" else scores)
        return self._json({
            "response": best, "done": True,
            "logprobs": [{"token": f" {best}", "logprob": -0.1,
                          "top_logprobs": top}],
        })


def start_server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


SRV, HOST = start_server()
PROMPT = ("Disease: Cystic fibrosis\nOptions:\nA. HBB\nB. CFTR\n"
          "C. GPR52\nD. GPR56\nE. None of the above\nAnswer:")

failures = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failures.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
    else:
        print(f"  ok    {name}")


def test_request_shape():
    RECEIVED.clear()
    OllamaBackend("gemma3:27b", host=HOST).label_logprobs(PROMPT, LABELS)
    p = RECEIVED[-1]
    assert p["model"] == "gemma3:27b"
    assert p["raw"] is True, "must bypass the chat template"
    assert p["stream"] is False
    assert p["logprobs"] is True
    assert p["top_logprobs"] == 20
    assert p["options"]["num_predict"] == 1, "one token is all stage 2 needs"
    assert p["options"]["temperature"] == 0
    assert p["prompt"].rstrip().endswith("Answer:")


def test_reads_the_right_letter():
    MODE["top"] = "list"
    got = OllamaBackend("gemma3:27b", host=HOST).label_logprobs(PROMPT, LABELS)
    assert all(v is not None for v in got.values()), got
    assert max(got, key=got.get) == "B", got   # B is CFTR in PROMPT


def test_dict_shaped_top_logprobs():
    MODE["top"] = "dict"
    try:
        got = OllamaBackend("gemma3:27b", host=HOST).label_logprobs(PROMPT, LABELS)
        assert max(got, key=got.get) == "B", got
    finally:
        MODE["top"] = "list"


def test_missing_label_is_none_not_invented():
    MODE["labels"] = set("ABC")
    try:
        got = OllamaBackend("gemma3:27b", host=HOST).label_logprobs(PROMPT, LABELS)
        assert got["D"] is None and got["E"] is None, got
        assert got["B"] is not None
    finally:
        MODE["labels"] = set("ABCDE")


def test_no_logprobs_is_a_clear_error():
    MODE["logprobs"] = False
    try:
        OllamaBackend("gemma3:27b", host=HOST).label_logprobs(PROMPT, LABELS)
    except BackendError as e:
        assert "logprobs" in str(e) and "0.12.11" in str(e)
    else:
        raise AssertionError("silently accepted a response with no logprobs")
    finally:
        MODE["logprobs"] = True


def test_unreachable_host_names_the_fix():
    try:
        OllamaBackend("m", host="http://127.0.0.1:1").version()
    except BackendError as e:
        assert "ollama serve" in str(e)
    else:
        raise AssertionError("unreachable host did not raise")


def test_probe_reports_capabilities():
    r = OllamaBackend("gemma3:27b", host=HOST).probe(LABELS)
    assert r["reachable"] and r["version"] == "0.12.11"
    assert r["model_present"] is True
    assert r["logprobs"] is True
    assert r["supports_pmi"] is False, "Ollama must never claim stage 1"


def test_pmi_refused_with_the_alternatives_named():
    try:
        OllamaBackend("gemma3:27b", host=HOST).score_continuations("Gene: ",
                                                                   ["CFTR"])
    except ScoringUnsupported as e:
        msg = str(e)
        assert "llamacpp" in msg and "transformers" in msg
        assert "frequency" in msg
    else:
        raise AssertionError("Ollama claimed it could score a supplied string")


def test_gguf_lookup_from_ollama_store():
    with tempfile.TemporaryDirectory() as root:
        man = os.path.join(root, "manifests", "registry.ollama.ai", "library",
                           "gemma3", "27b")
        os.makedirs(os.path.dirname(man), exist_ok=True)
        digest = "sha256:" + "ab" * 32
        with open(man, "w") as f:
            json.dump({"layers": [
                {"mediaType": "application/vnd.ollama.image.license",
                 "digest": "sha256:" + "cd" * 32},
                {"mediaType": "application/vnd.ollama.image.model",
                 "digest": digest},
            ]}, f)
        blobs = os.path.join(root, "blobs")
        os.makedirs(blobs, exist_ok=True)
        blob = os.path.join(blobs, digest.replace(":", "-"))
        open(blob, "wb").write(b"GGUF")
        assert resolve_ollama_gguf("gemma3:27b", root) == blob
        assert resolve_ollama_gguf("gemma3:27b", root) == \
            resolve_ollama_gguf("gemma3:27b", root)

        try:
            resolve_ollama_gguf("nope:latest", root)
        except BackendError as e:
            assert "OLLAMA_MODELS" in str(e) and "ollama list" in str(e)
        else:
            raise AssertionError("missing manifest did not raise")


def test_match_labels_ignores_leading_space():
    assert match_labels({" A": -1.0, "B": -2.0}, ["A", "B", "C"]) == \
        {"A": -1.0, "B": -2.0, "C": None}
    # keep the best score when a letter arrives in two forms
    assert match_labels({" A": -3.0, "A": -1.0}, ["A"]) == {"A": -1.0}


GENES14 = [l.strip() for l in
           open(os.path.join(HERE, "..", "examples", "candidates.txt"))
           if l.strip() and not l.startswith("#")]


def test_ranker_degrades_loudly_on_ollama():
    r = GeneRanker("gemma3:27b", backend="ollama", host=HOST)
    out = r.rank("Cystic fibrosis", GENES14)
    assert out["backend"] == "ollama"
    assert out["stage1"] is None, "PMI must not be claimed on Ollama"
    assert out["plan"]["stages"] == ["labels"]
    assert out["warnings"], "degradation must be reported, not silent"
    assert "frequency" in out["warnings"][0]
    assert {x["gene"] for x in out["stage2"]} == set(GENES14), \
        "every candidate must still be scored"
    assert out["call"] == "CFTR", out["call"]


def test_strict_refuses_instead_of_degrading():
    r = GeneRanker("gemma3:27b", backend="ollama", host=HOST, strict=True)
    try:
        r.rank("Cystic fibrosis", GENES14)
    except ScoringUnsupported:
        pass
    else:
        raise AssertionError("strict=True silently degraded")


def test_small_list_needs_no_degradation():
    r = GeneRanker("gemma3:27b", backend="ollama", host=HOST)
    out = r.rank("Cystic fibrosis", ["CFTR", "HBB", "GPR52", "APOE"])
    assert out["warnings"] == [], out["warnings"]
    assert out["call"] == "CFTR"




# --------------------------------------------------------------------------
# Docker. Ollama in a container is the common local setup, and it is where the
# GGUF stops being readable from the host - so the failure has to be diagnosed,
# not just reported as "not found".
# --------------------------------------------------------------------------

DOCKER_STUB = """#!/usr/bin/env python3
import json, os, sys
cfg = json.load(open(os.environ["DOCKER_STUB_CFG"]))
args = sys.argv[1:]
key = args[0] if args else ""
if key == "ps":
    sys.stdout.write(cfg.get("ps", ""))
elif key == "inspect":
    sys.stdout.write(cfg.get("inspect", ""))
elif key == "volume":
    sys.stdout.write(cfg.get("volume", ""))
elif key == "exec":
    sys.stdout.write(cfg.get("exec", ""))
else:
    sys.exit(1)
"""


class docker_stub:
    """Put a fake `docker` on PATH driven by a config dict."""

    def __init__(self, **cfg):
        self.cfg = cfg

    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        exe = os.path.join(self.dir, "docker")
        with open(exe, "w") as f:
            f.write(DOCKER_STUB)
        os.chmod(exe, 0o755)
        self.cfgfile = os.path.join(self.dir, "cfg.json")
        with open(self.cfgfile, "w") as f:
            json.dump(self.cfg, f)
        self.old_path = os.environ["PATH"]
        self.old_cfg = os.environ.get("DOCKER_STUB_CFG")
        os.environ["PATH"] = self.dir + os.pathsep + self.old_path
        os.environ["DOCKER_STUB_CFG"] = self.cfgfile
        return self

    def __exit__(self, *a):
        os.environ["PATH"] = self.old_path
        if self.old_cfg is None:
            os.environ.pop("DOCKER_STUB_CFG", None)
        else:
            os.environ["DOCKER_STUB_CFG"] = self.old_cfg


def _fake_store(root, digest_hex="ab" * 32):
    man = os.path.join(root, "models", "manifests", "registry.ollama.ai",
                       "library", "gemma3", "27b")
    os.makedirs(os.path.dirname(man), exist_ok=True)
    with open(man, "w") as f:
        json.dump({"layers": [{"mediaType": "application/vnd.ollama.image.model",
                               "digest": "sha256:" + digest_hex}]}, f)
    blobs = os.path.join(root, "models", "blobs")
    os.makedirs(blobs, exist_ok=True)
    blob = os.path.join(blobs, "sha256-" + digest_hex)
    open(blob, "wb").write(b"GGUF")
    return blob


def test_docker_container_detected():
    with docker_stub(ps="ollama\tollama/ollama:latest\nweb\tnginx\n"):
        assert docker_ollama_containers() == ["ollama"]


def test_docker_bind_mount_store_is_used():
    with tempfile.TemporaryDirectory() as home:
        blob = _fake_store(home)
        with docker_stub(ps="ollama\tollama/ollama\n",
                         inspect=f"/root/.ollama\t{home}\n"):
            assert docker_ollama_store() == os.path.join(home, "models")
            # and the resolver reaches it without OLLAMA_MODELS being set
            old = os.environ.pop("OLLAMA_MODELS", None)
            try:
                assert resolve_ollama_gguf("gemma3:27b") == blob
            finally:
                if old is not None:
                    os.environ["OLLAMA_MODELS"] = old


def test_named_volume_not_on_host_is_diagnosed():
    """The Docker Desktop case: container exists, files are unreachable."""
    digest = "ef" * 32
    manifest = json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model",
         "digest": "sha256:" + digest}]})
    with docker_stub(ps="ollama\tollama/ollama\n",
                     inspect="/root/.ollama\t/var/lib/docker/volumes/ollama/_data\n",
                     exec=manifest):
        assert docker_ollama_store() is None
        cmd = docker_cp_command("gemma3:27b")
        assert cmd.startswith("docker cp ollama:")
        assert f"blobs/sha256-{digest}" in cmd
        assert cmd.endswith("./gemma3-27b.gguf")

        with tempfile.TemporaryDirectory() as empty:
            old = os.environ.get("OLLAMA_MODELS")
            os.environ["OLLAMA_MODELS"] = empty
            try:
                resolve_ollama_gguf("gemma3:27b")
            except BackendError as e:
                msg = str(e)
                assert "Docker" in msg, msg
                assert "bind-mount" in msg or "~/.ollama:/root/.ollama" in msg
                assert "docker cp ollama:" in msg
            else:
                raise AssertionError("unreadable docker store was not diagnosed")
            finally:
                if old is None:
                    os.environ.pop("OLLAMA_MODELS", None)
                else:
                    os.environ["OLLAMA_MODELS"] = old


def test_host_discovery_finds_the_running_server():
    port = SRV.server_port
    found = discover_ollama_host([f"http://127.0.0.1:{port + 1}",
                                  f"http://127.0.0.1:{port}"])
    assert found == f"http://127.0.0.1:{port}", found
    assert discover_ollama_host(["http://127.0.0.1:1"]) is None


if __name__ == "__main__":
    print(f"backend tests (fake Ollama on {HOST})")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all passed")
