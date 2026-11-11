"""Audiobook library: scans a books folder (PDF/EPUB/MOBI/FB2/TXT) and reads it aloud with word timings: Kokoro for English, OmniVoice (omni_worker.py) for Bangla.

usage: ./run.sh [books_dir] [port]   (or: python library_server.py [books_dir] [port] [voice])
State (collections, progress), covers and parsed text live in ./data next to this script.
"""
import base64, collections, hashlib, io, json, os, re, subprocess, sys, threading
from concurrent.futures import Future
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ESPEAK_DATA_PATH", "/opt/homebrew/share/espeak-ng-data")
import numpy as np, pymupdf, soundfile as sf, torch
from kokoro import KPipeline
import cast

BOOKS = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Books")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
VOICE = sys.argv[3] if len(sys.argv) > 3 else "af_heart"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXTS = {".pdf", ".epub", ".mobi", ".azw3", ".fb2", ".txt"}
BN = re.compile(r"[ঀ-৿]")  # Bangla paragraphs go to OmniVoice instead of Kokoro
OMNI_PY = os.path.expanduser(os.environ.get("OMNIVOICE_PYTHON", os.path.join(HERE, ".venv-omni", "bin", "python")))
SR, AHEAD, CACHE_MAX = 24000, 4, 12  # audio lives in RAM only: ~4 ahead + a few behind
SKIP_TITLES = re.compile(r"^(cover|title|title page|copyright|contents|table of contents|dedication|also by|praise|half title)", re.I)
LLM = os.environ.get("READALONG_LLM", "qwen3.5:2b")  # Ollama model for speaker attribution
MALE = ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx", "am_adam"]
FEMALE = ["af_bella", "af_nicole", "af_sarah", "af_kore", "af_aoede", "af_nova", "af_sky", "af_river", "af_jessica", "af_alloy"]
for d in ("covers", "parsed", "pron", "speakers", "voices"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)


# ---------- library ----------
def book_id(rel):
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


def scan():
    books = {}
    for root, dirs, files in os.walk(BOOKS):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS and not f.startswith("."):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, BOOKS)
                books[book_id(rel)] = path
    return books


meta_cache = {}  # path -> (mtime, meta)


def meta(bid, path):
    mt = os.path.getmtime(path)
    if path in meta_cache and meta_cache[path][0] == mt:
        return meta_cache[path][1]
    stem = os.path.splitext(os.path.basename(path))[0]
    title, author = stem, ""
    if " -- " in stem:  # "Title -- Author -- ..." style filenames
        parts = stem.split(" -- ")
        title, author = parts[0], parts[1]
    else:
        try:
            with pymupdf.open(path) as d:
                m = d.metadata or {}
            if (m.get("title") or "").strip():
                title = m["title"].strip()
            author = (m.get("author") or "").strip()
        except Exception:
            pass
    title = title.replace("_", ":").strip()
    m = {"id": bid, "title": title, "author": author, "format": os.path.splitext(path)[1][1:].lower(),
         "file": os.path.relpath(path, BOOKS), "added": mt}
    meta_cache[path] = (mt, m)
    return m


def cover(bid, path):
    out = os.path.join(DATA, "covers", bid + ".jpg")
    if not os.path.exists(out):
        with pymupdf.open(path) as d:
            page = d[0]
            z = 240 / max(page.rect.width, 1)
            page.get_pixmap(matrix=pymupdf.Matrix(z, z)).save(out, jpg_quality=70)
    return open(out, "rb").read()


# ---------- text extraction ----------
TERMINAL = tuple(".!?:;\"”’)…—")


def clean(t):
    t = re.sub(r"-\n(?=[a-z])", "", t)  # de-hyphenate line breaks
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(?<=[A-Za-z.,;:!?’”)])\d{1,3}\b", "", t)  # footnote markers
    # small-caps openings ("DURING A BREAK IN THE") read badly; sentence-case them
    t = re.sub(r"^[A-Z’' .,]{6,}(?= )", lambda m: m.group(0).capitalize(), t)
    return t


def txt_chapters(path):
    """Plain text (e.g. OCR output): blank lines split paragraphs; a lone one- or two-word line is a chapter title."""
    chapters = [{"title": "Opening", "paras": []}]
    for t in map(clean, open(path, encoding="utf-8", errors="replace").read().split("\n\n")):
        if re.fullmatch(r"[ঀ-৿\w]+( [ঀ-৿\w]+)?", t):
            chapters.append({"title": t, "paras": []})
        elif t:
            chapters[-1]["paras"].append(t)
    return [c for c in chapters if c["paras"]]


def pdf_chapters(path):
    with pymupdf.open(path) as d:
        n = len(d)
        pages = [[b[4] for b in d[i].get_text("blocks") if b[6] == 0] for i in range(n)]
        toc = [(t, p - 1) for _, t, p in d.get_toc() if 0 < p <= n]
    # running headers/footers: short blocks repeated on many pages
    norm = lambda s: re.sub(r"\d+", "", s).strip().lower()
    freq = collections.Counter(norm(b) for pg in pages for b in set(pg) if len(b) < 80)
    rep = {k for k, v in freq.items() if v >= max(4, n // 8)}

    def paras(a, b):
        out = []
        for pg in pages[a:b]:
            for blk in pg:
                if norm(blk) in rep or re.fullmatch(r"\s*\d+\s*", blk):
                    continue
                t = clean(blk)
                if not t:
                    continue
                # join a paragraph split across a block/page break
                if out and not out[-1].endswith(TERMINAL) and t[:1].islower():
                    out[-1] += " " + t
                else:
                    out.append(t)
        return out

    starts = sorted({p for _, p in toc}) if len(toc) >= 2 else []
    if starts:
        titles = {}
        for t, p in toc:
            titles.setdefault(p, t.strip())
        bounds = ([(0, "Opening")] if starts[0] > 0 else []) + [(p, titles[p]) for p in starts]
    else:
        bounds = [(p, f"Pages {p + 1}–{min(p + 10, n)}") for p in range(0, n, 10)]
    chapters = []
    for i, (p, t) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else n
        ps = paras(p, end)
        if ps:
            chapters.append({"title": t, "paras": ps})
    return chapters


def parse(bid, path):
    mt = int(os.path.getmtime(path))
    cache_file = os.path.join(DATA, "parsed", f"{bid}-{mt}.json")
    if os.path.exists(cache_file):
        return json.load(open(cache_file))
    chapters = txt_chapters(path) if path.lower().endswith(".txt") else pdf_chapters(path)
    text =" ".join(" ".join(c["paras"]) for c in chapters)
    letters = re.findall(r"[^\W\d_]", text)
    latin = sum(c.isascii() for c in letters) / max(len(letters), 1)
    start = next((i for i, c in enumerate(chapters) if not SKIP_TITLES.match(c["title"]) and len(" ".join(c["paras"])) > 1500), 0)
    res = {"chapters": chapters, "start": start, "english": latin > 0.8, "empty": not chapters}
    json.dump(res, open(cache_file, "w"))
    return res


parsed = {}


def get_book(bid):
    if bid not in parsed:
        parsed[bid] = parse(bid, scan()[bid])
    return parsed[bid]


# ---------- TTS scheduler: one GPU worker, urgent requests jump the queue ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = KPipeline(lang_code="a", device=device, repo_id="hexgrad/Kokoro-82M")
audio_cache = collections.OrderedDict()
pending, order, cond = {}, collections.deque(), threading.Condition()


# ---------- per-book pronunciation: data/pron/<bid>.json {"Watanabe": "/wˌɑtənˈɑbA/" or "wah tah nah bay"} ----------
pron_cache = {}  # bid -> (mtime, regex, {word: phonemes})


def mtime(f):
    try:
        return os.path.getmtime(f)
    except OSError:
        return 0


def pron(bid):
    f = os.path.join(DATA, "pron", bid + ".json")
    mt = mtime(f)
    if bid not in pron_cache or pron_cache[bid][0] != mt:
        d = json.load(open(f)) if mt else {}
        # "/.../" = Kokoro (misaki) phonemes; anything else is a respelling we run through G2P
        ph = {w.lower(): v.strip().strip("/") if v.strip().startswith("/") else "".join(pipe.g2p(v)[0].split()) for w, v in d.items()}
        rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b", re.I) if d else None
        pron_cache[bid] = (mt, rx, ph)
    return pron_cache[bid]


def apply_pron(bid, text):
    _, rx, ph = pron(bid)
    return rx.sub(lambda m: f"[{m.group(0)}](/{ph[m.group(0).lower()]}/)", text) if rx else text


# ---------- character voices: data/speakers/<bid>/<c>.json + data/voices/<bid>.json registry ----------
def cast_file(bid, c):
    return os.path.join(DATA, "speakers", bid, f"{c}.json")


def voices_file(bid):
    return os.path.join(DATA, "voices", bid + ".json")


def segments(bid, c, p, text):
    """[(text, voice)]: quotes in the speaker's voice, narration (and unsure speakers) in the narrator's."""
    try:
        who, conf = json.load(open(cast_file(bid, c))).get(str(p), [None, "low"])
        voice = json.load(open(voices_file(bid))).get(who) if who and conf != "low" else None
    except (OSError, ValueError):
        voice = None
    if not voice:
        return [(text, VOICE)]
    out, last = [], 0
    for m in cast.QUOTE.finditer(text):
        out += [(text[last:m.start()], VOICE)] if m.start() > last else []
        out.append((m.group(0), voice))
        last = m.end()
    return out + ([(text[last:], VOICE)] if last < len(text) else [])


def version(bid, c):  # cached audio is stale once pronunciations or the cast change
    return (mtime(os.path.join(DATA, "pron", bid + ".json")), mtime(cast_file(bid, c)), mtime(voices_file(bid)))


omni = None  # OmniVoice subprocess (own venv), started on the first Bangla paragraph


def omni_synth(text):
    global omni
    if omni is None or omni.poll() is not None:
        omni = subprocess.Popen([OMNI_PY, os.path.join(HERE, "omni_worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    omni.stdin.write(json.dumps({"text": text}) + "\n")
    omni.stdin.flush()
    line = omni.stdout.readline()
    if not line:
        raise RuntimeError("OmniVoice worker exited")
    return json.loads(line)


def synth(bid, c, p):
    text = get_book(bid)["chapters"][c]["paras"][p]
    if BN.search(text):
        return omni_synth(text)
    audio, words, offset = [], [], 0.0
    for seg, voice in segments(bid, c, p, text):
        if words and seg[:1].isspace() and not words[-1]["w"][-1:].isspace():
            words[-1]["w"] += " "
        for r in pipe(apply_pron(bid, seg), voice=voice, speed=1.0):
            a = r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio)
            for t in r.tokens or []:
                # glue punctuation/quotes to the neighbouring word so each highlight is a whole word
                if words and (t.start_ts is None or not words[-1]["w"][-1:].isspace()):
                    words[-1]["w"] += t.text + t.whitespace
                elif t.start_ts is not None:
                    words.append({"w": t.text + t.whitespace, "s": offset + t.start_ts, "e": offset + t.end_ts})
            audio.append(a)
            offset += len(a) / SR
        if words and seg[-1:].isspace() and not words[-1]["w"][-1:].isspace():
            words[-1]["w"] += " "
    if not audio:  # nothing speakable (e.g. a lone symbol)
        audio = [np.zeros(SR // 4, dtype=np.float32)]
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(audio), SR, format="WAV")
    return {"words": words, "audio": base64.b64encode(buf.getvalue()).decode()}


def request(key, urgent):
    with cond:
        if key in audio_cache:
            audio_cache.move_to_end(key)
            f = Future(); f.set_result(audio_cache[key]); return f
        if key not in pending:
            pending[key] = Future()
            order.append(key)
        if urgent and key in order:
            order.remove(key); order.appendleft(key)
        cond.notify()
        return pending[key]


def worker():
    while True:
        with cond:
            while not order:
                cond.wait()
            key = order.popleft()
        try:
            res = synth(*key[:3])
            with cond:
                audio_cache[key] = res
                while len(audio_cache) > CACHE_MAX:
                    audio_cache.popitem(last=False)
            pending.pop(key).set_result(res)
        except Exception as e:
            pending.pop(key).set_exception(e)


def prefetch(bid, c, p):
    chs = get_book(bid)["chapters"]
    for _ in range(AHEAD):
        p += 1
        if p >= len(chs[c]["paras"]):
            c, p = c + 1, 0
            if c >= len(chs):
                return
        request((bid, c, p, version(bid, c)), False)


# ---------- speaker attribution: one chapter at a time, off the audio path ----------
attr_todo, attr_done, attr_cond = collections.deque(), set(), threading.Condition()


def want_cast(bid, c):
    b = get_book(bid)
    if not b["english"] or not 0 <= c < len(b["chapters"]) or os.path.exists(cast_file(bid, c)):
        return
    with attr_cond:
        if (bid, c) not in attr_done:
            attr_done.add((bid, c))
            attr_todo.append((bid, c))
            attr_cond.notify()


def attr_worker():
    while True:
        with attr_cond:
            while not attr_todo:
                attr_cond.wait()
            bid, c = attr_todo.popleft()
        try:
            speakers, genders = cast.attribute(get_book(bid)["chapters"][c]["paras"], LLM)
            reg = json.load(open(voices_file(bid))) if os.path.exists(voices_file(bid)) else {}
            for name, g in genders.items():  # a character keeps the same voice for the whole book
                if name not in reg:
                    pool = [v for v in (MALE if g == "m" else FEMALE if g == "f" else MALE + FEMALE) if v != VOICE]
                    free = [v for v in pool if v not in reg.values()] or pool
                    reg[name] = free[0]
            for f, d in ((voices_file(bid), reg), (cast_file(bid, c), speakers)):
                os.makedirs(os.path.dirname(f), exist_ok=True)
                json.dump(d, open(f + ".tmp", "w"), ensure_ascii=False, indent=1)
                os.replace(f + ".tmp", f)
            print(f"cast {bid} ch{c}: {sum(v[1] != 'low' for v in speakers.values())}/{len(speakers)} dialogue paragraphs voiced", flush=True)
        except Exception as e:
            print(f"cast {bid} ch{c} failed: {e}", flush=True)


# ---------- state ----------
STATE = os.path.join(DATA, "state.json")
state_lock = threading.Lock()


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"collections": {}, "progress": {}}


# ---------- HTTP ----------
class H(BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self.send(open(os.path.join(HERE, "library.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/api/library":
                self.send(json.dumps(sorted((meta(i, p) for i, p in scan().items()), key=lambda m: m["title"].lower())))
            elif u.path == "/api/state":
                self.send(json.dumps(load_state()))
            elif u.path == "/api/cover":
                self.send(cover(q["id"], scan()[q["id"]]), "image/jpeg")
            elif u.path == "/api/book":
                self.send(json.dumps(get_book(q["id"])))
            elif u.path == "/api/pron":
                get_book(q["id"])  # 404s unknown ids
                f = os.path.join(DATA, "pron", q["id"] + ".json")
                self.send(open(f).read() if os.path.exists(f) else "{}")
            elif u.path == "/api/para":
                key = (q["id"], int(q["c"]), int(q["p"]))
                res = request(key + (version(*key[:2]),), True).result()
                prefetch(*key)
                want_cast(key[0], key[1]); want_cast(key[0], key[1] + 1)
                self.send(json.dumps(res))
            else:
                self.send("not found", "text/plain", 404)
        except KeyError:
            self.send('{"error":"unknown book"}', code=404)
        except Exception as e:
            self.send(json.dumps({"error": str(e)}), code=500)

    def do_POST(self):
        u = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/api/state":
            try:
                s = json.loads(body)
                assert isinstance(s.get("collections"), dict) and isinstance(s.get("progress"), dict)
            except Exception:
                return self.send('{"error":"bad state"}', code=400)
            with state_lock:
                tmp = STATE + ".tmp"
                json.dump(s, open(tmp, "w"))
                os.replace(tmp, STATE)  # atomic: never leave a half-written state file
            self.send('{"ok":true}')
        elif u.path == "/api/pron":
            bid = parse_qs(u.query).get("id", [""])[0]
            try:
                d = json.loads(body)
                assert bid in scan() and isinstance(d, dict) and all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in d.items())
            except Exception:
                return self.send('{"error":"bad pronunciations"}', code=400)
            f = os.path.join(DATA, "pron", bid + ".json")
            json.dump(d, open(f + ".tmp", "w"), ensure_ascii=False, indent=1)
            os.replace(f + ".tmp", f)
            self.send('{"ok":true}')
        elif u.path == "/api/upload":
            name = os.path.basename(unquote(parse_qs(u.query).get("name", [""])[0])).strip()
            stem, ext = os.path.splitext(name)
            if ext.lower() not in EXTS or not stem or not body:
                return self.send('{"error":"unsupported file"}', code=400)
            dest, i = os.path.join(BOOKS, name), 1
            while os.path.exists(dest):
                dest, i = os.path.join(BOOKS, f"{stem} ({i}){ext}"), i + 1
            open(dest, "wb").write(body)
            self.send(json.dumps({"id": book_id(os.path.relpath(dest, BOOKS))}))
        else:
            self.send("not found", "text/plain", 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=attr_worker, daemon=True).start()
    print(f"Library: {BOOKS} ({len(scan())} books). Kokoro on {device}, voice {VOICE}.")
    print(f"Open http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
