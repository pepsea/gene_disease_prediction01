#!/usr/bin/env python3
"""Model backends: where the numbers actually come from.

The pipeline needs two different things from a model, and they are not equally
easy to get:

  A. score_continuations - the log probability of a string you SUPPLY.
     This is what stage 1 (PMI) runs on. The gene symbol is an input, so
     GPR52 and GPR56 get separate independent scores.

  B. label_logprobs - the log probabilities of the next token, restricted to
     A-E. This is what stage 2 runs on. The model emits one letter.

(B) is ordinary generation with logprobs turned on. (A) is not generation at
all - it is teacher forcing - and several local runtimes cannot do it.

    backend      A (PMI)  B (labels)  notes
    transformers   yes       yes      the reference implementation
    llamacpp       yes       yes      GGUF; can read Ollama's own blobs
    ollama         NO        yes      the API returns logprobs for generated
                                      tokens only; there is no echo or scoring
                                      endpoint, so a supplied symbol cannot be
                                      scored

Pick a backend with `make_backend()`, and check `supports_pmi` before assuming
stage 1 is available.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BackendError(RuntimeError):
    pass


class ScoringUnsupported(BackendError):
    """Raised when a backend cannot score a supplied string (stage 1)."""


def match_labels(candidates, labels):
    """Map a runtime's token->logprob dict onto the labels A-E.

    Runtimes disagree about leading spaces and about whether a label arrives as
    its own token, so match on the stripped text rather than assuming a form.
    Labels that never appear are returned as None; the caller decides what to do
    with a missing one rather than having a fabricated score pushed at it.
    """
    norm = {}
    for tok, lp in candidates.items():
        key = tok.strip()
        # keep the best logprob seen for a given letter
        if key and (key not in norm or lp > norm[key]):
            norm[key] = lp
    return {lab: norm.get(lab) for lab in labels}


class Backend:
    name = "base"
    supports_pmi = False

    def score_continuations(self, prefix, targets, batch_size=32):
        raise ScoringUnsupported(
            f"backend '{self.name}' cannot score supplied strings")

    def label_logprobs(self, prompt, labels):
        raise NotImplementedError

    def describe(self):
        return {"backend": self.name, "supports_pmi": self.supports_pmi}


class TransformersBackend(Backend):
    """Hugging Face transformers. Full support; the reference implementation."""

    name = "transformers"
    supports_pmi = True

    def __init__(self, model, dtype="bfloat16", device_map="auto"):
        from score_pmi import Scorer
        self.model_id = model
        self.scorer = Scorer(model, dtype=dtype, device_map=device_map)
        self.tok = self.scorer.tok
        # One set of weights, used by both stages.
        self.label_ids = {}

    def score_continuations(self, prefix, targets, batch_size=32):
        return self.scorer.score(prefix, list(targets), batch_size)

    def _ids_for(self, labels):
        key = tuple(labels)
        if key not in self.label_ids:
            ids = [self.tok.encode(" " + lab, add_special_tokens=False)[-1]
                   for lab in labels]
            if len(set(ids)) != len(labels):
                raise BackendError(
                    "Label tokens collide for this tokenizer. Run "
                    "check_tokenizer.py and pick different labels.")
            self.label_ids[key] = ids
        return self.label_ids[key]

    def label_logprobs(self, prompt, labels):
        import torch
        ids = self._ids_for(labels)
        enc = self.tok(prompt, return_tensors="pt").to(self.scorer.model.device)
        with torch.no_grad():
            logits = self.scorer.model(**enc).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return {lab: lp[i].item() for lab, i in zip(labels, ids)}

    def describe(self):
        d = super().describe()
        d["model"] = self.model_id
        return d


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _normalize_host(host):
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


# Where a local Ollama might be listening. The last two matter when Ollama is
# in Docker and the caller is in another container: "localhost" then means that
# container itself, not the host. host.docker.internal is Docker Desktop's name
# for the host; 172.17.0.1 is the default bridge gateway on Linux.
HOST_CANDIDATES = [
    "http://localhost:11434",
    "http://127.0.0.1:11434",
    "http://host.docker.internal:11434",
    "http://172.17.0.1:11434",
]


def discover_ollama_host(candidates=None, timeout=2):
    """First host that answers /api/version, or None."""
    for h in (candidates or HOST_CANDIDATES):
        try:
            OllamaBackend("", host=h, timeout=timeout).version()
            return _normalize_host(h)
        except BackendError:
            continue
    return None


class OllamaBackend(Backend):
    """Local Ollama over its native HTTP API.

    Stage 2 only. Ollama's `logprobs` / `top_logprobs` apply to tokens the model
    GENERATES; there is no echo option and no scoring endpoint, so the log
    probability of a supplied gene symbol cannot be obtained. That rules out
    stage 1. See `check_backend.py` for a probe against a running server, and
    LlamaCppBackend for the same weights with scoring available.

    Needs no extra packages - stdlib HTTP only.
    """

    name = "ollama"
    supports_pmi = False

    def __init__(self, model, host=DEFAULT_OLLAMA_HOST, top_logprobs=20,
                 timeout=120, options=None):
        self.model_id = model
        self.host = _normalize_host(host)
        self.top_logprobs = top_logprobs
        self.timeout = timeout
        self.options = {"temperature": 0, "num_predict": 1}
        if options:
            self.options.update(options)

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            raise BackendError(
                f"Ollama returned HTTP {e.code} for {path}: {body}") from e
        except urllib.error.URLError as e:
            raise BackendError(
                f"cannot reach Ollama at {self.host} ({e.reason}). "
                f"Is it running?  ollama serve") from e

    def version(self):
        req = urllib.request.Request(self.host + "/api/version")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode()).get("version")
        except Exception as e:
            raise BackendError(
                f"cannot reach Ollama at {self.host} ({e}). "
                f"Is it running?  ollama serve") from e

    def list_models(self):
        req = urllib.request.Request(self.host + "/api/tags")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        return [m.get("name") for m in data.get("models", [])]

    def _generate(self, prompt, num_predict=1):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            # raw: skip the chat template. These prompts are plain completions
            # that must end exactly at "Answer:".
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "options": dict(self.options, num_predict=num_predict),
        }
        return self._post("/api/generate", payload)

    @staticmethod
    def _first_token_candidates(data):
        """Pull {token: logprob} for the first generated position.

        Tolerates both shapes seen in the wild for top_logprobs: a list of
        {"token","logprob"} objects, and a plain {token: logprob} mapping.
        """
        lp = data.get("logprobs")
        if not lp:
            raise BackendError(
                "Ollama returned no logprobs. This needs a server new enough to "
                "support the logprobs/top_logprobs parameters (v0.12.11+), and "
                "a local model - Ollama Cloud has been reported to return null. "
                "Run check_backend.py for a full report.")
        first = lp[0]
        out = {}
        tok, val = first.get("token"), first.get("logprob")
        if tok is not None and val is not None:
            out[tok] = float(val)
        top = first.get("top_logprobs") or []
        if isinstance(top, dict):
            for t, v in top.items():
                out[t] = float(v)
        else:
            for alt in top:
                t, v = alt.get("token"), alt.get("logprob")
                if t is not None and v is not None:
                    out[t] = float(v)
        return out

    def label_logprobs(self, prompt, labels):
        data = self._generate(prompt, num_predict=1)
        return match_labels(self._first_token_candidates(data), labels)

    def probe(self, labels=("A", "B", "C", "D", "E")):
        """Report what this server can actually do, without guessing."""
        report = {"host": self.host, "model": self.model_id,
                  "reachable": False, "version": None, "model_present": None,
                  "logprobs": False, "labels_found": [], "labels_missing": [],
                  "supports_pmi": False, "error": None}
        try:
            report["version"] = self.version()
            report["reachable"] = True
        except BackendError as e:
            report["error"] = str(e)
            return report
        try:
            names = self.list_models()
            report["model_present"] = any(
                n == self.model_id or n.split(":")[0] == self.model_id.split(":")[0]
                for n in names)
            report["available_models"] = names
        except Exception:
            pass
        try:
            got = self.label_logprobs(
                "Disease: Cystic fibrosis\nOptions:\nA. HBB\nB. CFTR\n"
                "C. None of the above\nAnswer:", list(labels))
            report["logprobs"] = any(v is not None for v in got.values())
            report["labels_found"] = [k for k, v in got.items() if v is not None]
            report["labels_missing"] = [k for k, v in got.items() if v is None]
        except BackendError as e:
            report["error"] = str(e)
        return report

    def score_continuations(self, prefix, targets, batch_size=32):
        raise ScoringUnsupported(
            "Ollama cannot score a supplied string, so stage 1 (PMI) is not "
            "available through it: its logprobs cover generated tokens only, "
            "and there is no echo or evaluate endpoint.\n"
            "Three ways forward:\n"
            "  1. backend='llamacpp' - same GGUF weights Ollama already has on "
            "disk, scoring available. See resolve_ollama_gguf().\n"
            "  2. backend='transformers' - the reference path, needs the "
            "unquantized weights.\n"
            "  3. stage 2 alone, over all candidates. Works, and never "
            "generates a symbol, but without the PMI subtraction there is "
            "nothing removing corpus-frequency bias - check the frequency-only "
            "control before trusting it.")

    def describe(self):
        d = super().describe()
        d.update({"model": self.model_id, "host": self.host})
        return d


# --------------------------------------------------------------------------
# llama.cpp (GGUF) - reads the weights Ollama has already downloaded
# --------------------------------------------------------------------------

def _docker(*args, timeout=20):
    """Run a docker command. Returns stdout, or None if docker is unusable."""
    if not shutil.which("docker"):
        return None
    try:
        r = subprocess.run(("docker",) + args, capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def docker_ollama_containers():
    """Names of running containers that look like Ollama."""
    out = _docker("ps", "--format", "{{.Names}}\t{{.Image}}")
    if not out:
        return []
    names = []
    for line in out.splitlines():
        name, _, image = line.partition("\t")
        if "ollama" in image.lower() or "ollama" in name.lower():
            names.append(name)
    return names


def docker_ollama_store(container=None):
    """Host-side path of a dockerised Ollama's model store, or None.

    Two mount styles behave very differently here:

      -v ~/.ollama:/root/.ollama   bind mount; the files are ordinary host
                                   files and llama.cpp can read them in place.
      -v ollama:/root/.ollama      named volume; on Linux the data sits under
                                   /var/lib/docker/volumes/... and usually
                                   needs root, and on Docker Desktop it lives
                                   inside a VM and is not on the host at all.

    Returns a readable path only. When it returns None the weights exist but
    are not reachable from here - see docker_cp_command().
    """
    container = container or next(iter(docker_ollama_containers()), None)
    if container:
        out = _docker("inspect", container, "--format",
                      "{{range .Mounts}}{{.Destination}}\t{{.Source}}\n{{end}}")
        for line in (out or "").splitlines():
            dest, _, src = line.partition("\t")
            if dest.rstrip("/").endswith(".ollama") and src:
                path = os.path.join(src.strip(), "models")
                if os.path.isdir(path):
                    return path
    out = _docker("volume", "inspect", "ollama", "--format", "{{.Mountpoint}}")
    if out and out.strip():
        path = os.path.join(out.strip(), "models")
        if os.path.isdir(path):
            return path
    return None


def docker_manifest_digest(name, container=None):
    """Read the model layer digest from inside the container."""
    container = container or next(iter(docker_ollama_containers()), None)
    if not container:
        return None, None
    rel = _manifest_relpath(name)
    out = _docker("exec", container, "cat", "/root/.ollama/models/" + rel)
    if not out:
        return container, None
    try:
        manifest = json.loads(out)
    except json.JSONDecodeError:
        return container, None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            return container, layer["digest"].replace(":", "-")
    return container, None


def docker_cp_command(name, container=None):
    """The exact command to pull the GGUF out of a dockerised Ollama.

    Copying duplicates the file - a 27B Q4 is around 17GB - so bind-mounting
    ~/.ollama into the container instead is usually the better trade.
    """
    container, digest = docker_manifest_digest(name, container)
    if not container:
        return None
    safe = name.replace(":", "-").replace("/", "-")
    if not digest:
        return (f"# could not read the manifest for {name} inside {container}\n"
                f"docker exec {container} ollama list")
    return (f"docker cp {container}:/root/.ollama/models/blobs/{digest} "
            f"./{safe}.gguf")


def ollama_models_root():
    return os.environ.get("OLLAMA_MODELS",
                          os.path.expanduser("~/.ollama/models"))


def _manifest_relpath(name):
    repo, tag = (name.rsplit(":", 1) if ":" in name else (name, "latest"))
    parts = repo.split("/")
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library"] + parts
    elif len(parts) == 2:
        parts = ["registry.ollama.ai"] + parts
    return "/".join(["manifests"] + parts + [tag])


def _manifest_candidates(name, root):
    yield os.path.join(root, *_manifest_relpath(name).split("/"))


def resolve_ollama_gguf(name, root=None, allow_docker=True):
    """Find the GGUF file behind an Ollama model name.

    Ollama stores a manifest per model:tag listing content-addressed blobs. The
    weights are the layer with mediaType application/vnd.ollama.image.model.
    Pointing llama.cpp at that blob reuses the download instead of fetching the
    same quantization twice.

    Returns the blob path, or raises with what it looked for.
    """
    roots = [root] if root else [ollama_models_root()]
    if root is None and allow_docker:
        docker_store = docker_ollama_store()
        if docker_store and docker_store not in roots:
            roots.append(docker_store)
    tried = []
    for r in roots:
        try:
            return _resolve_in_root(name, r, tried)
        except _NotHere:
            continue
    hint = ""
    if allow_docker and docker_ollama_containers():
        cmd = docker_cp_command(name)
        hint = ("\n\nOllama appears to be running in Docker. Its model store is "
                "not readable from here, which happens with a named volume "
                "(-v ollama:/root/.ollama) and always on Docker Desktop, where "
                "the volume lives inside a VM.\nEither bind-mount it instead "
                "(-v ~/.ollama:/root/.ollama) and restart the container, or "
                "copy the file out:\n  " + (cmd or "docker ps"))
    raise BackendError(
        f"no Ollama manifest for {name!r}. Looked in:\n  " +
        "\n  ".join(tried) +
        f"\nSet OLLAMA_MODELS if your store is elsewhere, or pass an explicit "
        f"model_path. `ollama list` shows the names that exist." + hint)


class _NotHere(Exception):
    pass


def _resolve_in_root(name, root, tried):
    for man in _manifest_candidates(name, root):
        tried.append(man)
        if not os.path.exists(man):
            continue
        with open(man) as f:
            manifest = json.load(f)
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = layer["digest"].replace(":", "-")
                blob = os.path.join(root, "blobs", digest)
                if not os.path.exists(blob):
                    raise BackendError(
                        f"manifest for {name} points at a missing blob: {blob}")
                return blob
        raise BackendError(f"no model layer in the manifest for {name}: {man}")
    raise _NotHere()


class LlamaCppBackend(Backend):
    """llama-cpp-python over a GGUF file. Both stages available.

    This is the recommended local path when the weights come from Ollama:
    same quantized file, but scoring works because llama.cpp will evaluate a
    supplied token sequence instead of only generating.

        pip install llama-cpp-python

    Prompts are passed as token ids rather than text, so the prefix/target
    boundary is exact by construction and cannot be shifted by a subword merge
    straddling the join.
    """

    name = "llamacpp"
    supports_pmi = True

    def __init__(self, model_path=None, ollama_model=None, n_ctx=4096,
                 n_gpu_layers=-1, verbose=False, **kw):
        if model_path is None:
            if ollama_model is None:
                raise BackendError("give model_path= or ollama_model=")
            model_path = resolve_ollama_gguf(ollama_model)
        self.model_path = model_path
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise BackendError(
                "llama-cpp-python is not installed:  pip install llama-cpp-python"
            ) from e
        # logits_all is what makes per-token scoring of the prompt possible.
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, logits_all=True,
                         verbose=verbose, **kw)

    def _tok(self, text, add_bos=False):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos,
                                 special=False)

    def score_continuations(self, prefix, targets, batch_size=32):
        import numpy as np
        prefix_ids = self._tok(prefix, add_bos=True)
        out = np.zeros(len(targets), dtype=np.float64)
        for i, t in enumerate(targets):
            target_ids = self._tok(t, add_bos=False)
            ids = prefix_ids + target_ids
            res = self.llm.create_completion(ids, max_tokens=0, logprobs=1,
                                             echo=True, temperature=0.0)
            lp = res["choices"][0]["logprobs"]["token_logprobs"]
            # token_logprobs is aligned with the prompt tokens; the first has no
            # predecessor and comes back as None.
            tail = lp[len(prefix_ids):len(ids)]
            out[i] = float(sum(x for x in tail if x is not None))
        return out

    def label_logprobs(self, prompt, labels):
        res = self.llm.create_completion(prompt, max_tokens=1, logprobs=20,
                                         temperature=0.0)
        info = res["choices"][0].get("logprobs") or {}
        tops = info.get("top_logprobs") or [{}]
        cand = dict(tops[0] or {})
        toks = info.get("tokens") or []
        tlp = info.get("token_logprobs") or []
        if toks and tlp and tlp[0] is not None:
            cand.setdefault(toks[0], tlp[0])
        return match_labels({k: float(v) for k, v in cand.items()}, labels)

    def describe(self):
        d = super().describe()
        d["model_path"] = self.model_path
        return d


def make_backend(model, backend="transformers", **kw):
    """backend: 'transformers' | 'ollama' | 'llamacpp'."""
    backend = (backend or "transformers").lower()
    if backend == "transformers":
        return TransformersBackend(model, **kw)
    if backend == "ollama":
        return OllamaBackend(model, **kw)
    if backend in ("llamacpp", "llama.cpp", "gguf"):
        # A path means a GGUF file directly; anything else is an Ollama model
        # name whose blob we look up.
        if os.path.exists(str(model)):
            return LlamaCppBackend(model_path=model, **kw)
        return LlamaCppBackend(ollama_model=model, **kw)
    raise BackendError(f"unknown backend {backend!r}")
