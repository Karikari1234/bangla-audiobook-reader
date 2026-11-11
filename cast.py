"""Who speaks each dialogue paragraph of a chapter.

Explicit tags ("said Gerasim", "Ivan Ilyich said") are trusted. A local LLM via Ollama fills in the rest.
A small LLM's own confidence is meaningless, so confidence comes from evidence instead:
  high   = explicit tag in the paragraph
  medium = LLM answer that names someone who appears in the chapter text
  low    = unknown / hallucinated name -> the reader keeps the narrator voice
All quotes in one paragraph are treated as one speaker (the usual convention in prose).
"""
import json, re, urllib.request

QUOTE = re.compile(r"“[^”]*(?:”|$)|\"[^\"]*(?:\"|$)")
VERBS = "said|asked|replied|answered|cried|shouted|whispered|muttered|exclaimed|added|continued|began|repeated|called|murmured|went on"
NAME = r"([A-Z][\w’'-]+(?: [A-Z][\w’'-]+){0,2})"
TAG = re.compile(rf"[”\"],? (?:{VERBS}) {NAME}|[”\"],? {NAME} (?:{VERBS})\b|[”\"],? (I) (?:{VERBS})\b")
NARRATOR = "(narrator)"  # first-person "I said": stays in the narrator's voice
NOT_NAMES = {"I", "He", "She", "They", "We", "You", "It", "The", "His", "Her"}
SPEECH = re.compile(rf"(?:^|[.!?]\s)\s*[“\"]|[”\"],? (?:\w+ ){{0,2}}(?:{VERBS})\b")  # skips scare quotes / titles like the “field well.”
WINDOW, CONTEXT, LOOKBACK = 25, 3, 15  # paragraphs per LLM call, untagged paragraphs shown before each window


def tagged(p):
    for m in TAG.finditer(p):
        n = m.group(1) or m.group(2) or m.group(3)
        if n == "I":
            return NARRATOR
        if n.split()[0] not in NOT_NAMES:
            return n


def norm(t):
    return re.sub(r"\W+", " ", t.replace("’", "'")).strip().lower()


def qnum(q, said):
    """Small models answer 3, "3", "[Q3]" or echo the quote's text; -> 1-based quote number or None."""
    m = re.fullmatch(r"\[?Q?(\d+)\]?", str(q).strip())
    if m:
        return int(m.group(1))
    t = norm(str(q))[:30]
    return next((k + 1 for k, s in enumerate(said) if t and s.startswith(t)), None)


def ask_llm(paras, lo, hi, model, url):
    lines, owner, said = [], [], []  # owner[k] = paragraph index of quote Q(k+1), said[k] = its text
    for i in range(max(0, lo - CONTEXT), hi):
        p = paras[i]
        if i >= lo and SPEECH.search(p):
            def mark(m, i=i):
                owner.append(i)
                said.append(norm(m.group(0)))
                return f"[Q{len(owner)}]{m.group(0)}"
            p = QUOTE.sub(mark, p)
        lines.append(p)
    if not owner:
        return {}
    prompt = ("Below is a passage from a novel. Each quotation is marked [Qn]. For every quotation, say who speaks it. "
              'If the first-person narrator ("I") speaks, answer "narrator".\n'
              'Reply with JSON: {"quotes": [{"q": 1, "speaker": "full character name or unknown", "gender": "m|f"}]}\n\n'
              + "\n\n".join(lines))
    body = json.dumps({"model": model, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    res = json.load(urllib.request.urlopen(urllib.request.Request(url, body), timeout=600))
    answers = {}
    for q in json.loads(res["message"]["content"]).get("quotes", []):
        k = qnum(q.get("q"), said)
        if k and 1 <= k <= len(owner) and isinstance(q.get("speaker"), str):
            answers.setdefault(k, (q["speaker"].strip(), q.get("gender")))
    out = {}
    for k in sorted(answers):
        out.setdefault(owner[k - 1], answers[k])  # first quote of the paragraph decides
    return out


def pronoun(p):
    m = re.search(rf"[”\"],? (?:(she|he) (?:{VERBS})|(?:{VERBS}) (she|he))\b", p)
    return m and (m.group(1) or m.group(2))


def attribute(paras, model="qwen3.5:2b", url="http://localhost:11434/api/chat"):
    """-> ({para index: [speaker, confidence]}, {speaker: 'm'|'f'|None})
    Order of trust: explicit tag > "she said" with one such character named nearby > alternation inside a
    back-and-forth anchored by those > LLM guess naming someone mentioned nearby > narrator voice."""
    llm = {}
    for lo in range(0, len(paras), WINDOW):
        if any(SPEECH.search(p) for p in paras[lo:lo + WINDOW]):
            try:
                llm.update(ask_llm(paras, lo, min(lo + WINDOW, len(paras)), model, url))
            except Exception:
                pass  # shortcut: no Ollama / bad JSON -> only tagged paragraphs get voices
    genders = {}
    for guess, g in llm.values():
        if guess and guess.lower() not in ("narrator", "i", "me", "unknown") and g in ("m", "f"):
            genders.setdefault(guess, g)
    near = lambda name, i: name in " ".join(paras[max(0, i - LOOKBACK):i + 1])
    speech = [i for i, p in enumerate(paras) if SPEECH.search(p)]
    anchor = {}
    for i in speech:
        name, pr = tagged(paras[i]), pronoun(paras[i])
        if name:
            anchor[i] = name
        elif pr:
            fits = [n for n, g in genders.items() if g == {"she": "f", "he": "m"}[pr] and near(n, i)]
            if len(fits) == 1:
                anchor[i] = fits[0]
    speakers = {}
    for run in _runs(speech):
        pair = {anchor[i] for i in run if i in anchor}
        for k, i in enumerate(run):
            if i in anchor:
                speakers[i] = [anchor[i], "high" if tagged(paras[i]) else "medium"]
                continue
            if len(pair) == 2:  # two people taking turns: untagged lines alternate
                j = min((j for j in range(len(run)) if run[j] in anchor), key=lambda j: (abs(j - k), j > k))
                a = anchor[run[j]]
                speakers[i] = [a if (k - j) % 2 == 0 else (pair - {a}).pop(), "medium"]
                continue
            guess = llm.get(i, (None, None))[0]
            guess = NARRATOR if guess and guess.lower() in ("narrator", "i", "me") else guess
            if guess == NARRATOR:
                speakers[i] = [NARRATOR, "medium"]
            elif (guess and guess.lower() != "unknown" and guess.split()[0] not in NOT_NAMES and near(guess, i)
                  and not re.match(rf"[“\"]{re.escape(guess)}\b", paras[i])):  # “Gerasim, …” is spoken *to* Gerasim
                speakers[i] = [guess, "medium"]
            else:
                speakers[i] = [None, "low"]
    genders = {s: genders.get(s) for s, _ in speakers.values() if s and s != NARRATOR}
    return speakers, genders


def _runs(idx):  # consecutive dialogue paragraphs; a narration-only paragraph ends the exchange
    out = []
    for i in idx:
        if out and out[-1][-1] == i - 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


if __name__ == "__main__":  # self-check of the deterministic part (no LLM needed)
    assert tagged("“Gerasim,” Ivan Ilyich said weakly.") == "Ivan Ilyich"
    assert tagged("“No sir, not at all,” said Gerasim, who had learned") == "Gerasim"
    assert tagged("“Gerasim,” he said to him, “are you busy now?”") is None
    assert tagged("“Fine,” I said.") == NARRATOR
    sp, g = attribute(["He came in.", "“Hello,” said Anna.", "“Hi.”"], url="http://127.0.0.1:9/none")
    assert sp == {1: ["Anna", "high"], 2: [None, "low"]} and g == {"Anna": None}, (sp, g)
    assert not SPEECH.search("Of course: the “field well.” I have no idea")
    assert SPEECH.search("Naoko took my hand. “Don’t you worry,” she said.")
    assert SPEECH.search("Naoko smiled and shook her head. “No, two will be enough.”")
    sp, _ = attribute(["“Hi,” said Anna.", "“Hello.”", "“How are you?”", "“Fine,” I said.", "“Good.”"], url="http://127.0.0.1:9/none")
    assert [sp[i][0] for i in range(5)] == ["Anna", NARRATOR, "Anna", NARRATOR, "Anna"], sp
    assert qnum(3, []) == 3 and qnum("[Q2]", []) == 2 and qnum("Wrong how?", ["hi", "wrong how"]) == 2
    print("ok")
