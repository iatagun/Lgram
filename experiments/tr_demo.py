"""
Throwaway local UI for the Turkish coreference model + centering.

Shows, for a text you type: the sentences, the pronoun placeholders lgram.tr inserts
for dropped subjects / implicit possessors, the model's coreference clusters (colours),
and per sentence the Cf ranking, Cb, Cp and transition — model and rules side by side.
Hover a word for its DizgeBERT analysis.

Usage:
    python experiments/tr_demo.py MODEL_DIR [--port 8765]
then open http://127.0.0.1:8765  (local only; standard library server, no extra deps)
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tr_coref_centering import build_stream, cluster_words, slot_index  # noqa: E402

from lgram.tr.centering import analyze_parsed  # noqa: E402
from lgram.tr.parser import JointParser, split_sentences  # noqa: E402

SPEAKERS = {"@1sg": "ben (konuşan)", "@2sg": "sen (dinleyen)",
            "@1pl": "biz", "@2pl": "siz"}  # fmt: skip
LOCK = threading.Lock()
PARSER = MODEL = None


def analyze(text: str) -> dict:
    return analyze_sentences(split_sentences(text))


def analyze_sentences(sentences) -> dict:
    """Rules and (when MODEL is loaded) model analysis of already split sentences."""
    pairs = [(s, PARSER.parse(s)) for s in sentences]
    pairs = [(s, p) for s, p in pairs if p]
    sents, parses = [s for s, _ in pairs], [p for _, p in pairs]
    if not parses:
        return {"sentences": [], "entities": {}}
    stream, where = build_stream(parses)
    clusters = cluster_words(MODEL, stream) if MODEL is not None else {}

    # stream items, with the parse of real tokens for the tooltip
    items = [None] * len(stream)
    for (i, kind, pos), idx in where.items():
        item = {"i": i, "form": stream[idx], "kind": kind, "cluster": clusters.get(idx),
                "noun": False}  # fmt: skip
        if kind == "tok":
            t = parses[i][pos - 1]
            feats = " ".join(f"{k}={v}" for k, v in t.feats.items())
            item["info"] = f"{t.upos} · {t.deprel} → {t.head} · {feats}"
            item["noun"] = t.upos in ("NOUN", "PROPN")
        items[idx] = item
    # a cluster is named after its first noun; a cluster of pronouns only after a
    # placeholder's plain form ("ben", "o"), else after its first word
    names = {}
    for ok in (
        lambda it: it["noun"],
        lambda it: it["kind"] != "tok",
        lambda it: True,
    ):
        for it in items:
            if it["cluster"] and ok(it):
                names.setdefault(it["cluster"], it["form"])

    def label(key):
        return SPEAKERS.get(key) or names.get(key) or key

    def report(rep):
        return [
            {
                "transition": u.transition.value,
                "cb": label(u.cb) if u.cb else None,
                "cp": label(u.cp) if u.cp else None,
                "cf": [
                    {"key": m.key, "label": label(m.key), "kind": m.kind} for m in u.cf
                ],
                "unresolved": u.unresolved,
            }
            for u in rep.utterances
        ]

    rules = report(analyze_parsed(sents, parses))
    if MODEL is None:
        model = [None] * len(sents)
    else:
        model = report(
            analyze_parsed(
                sents,
                parses,
                identity=lambda i, m: clusters.get(slot_index(where, i, m)),
            )
        )
    return {
        "sentences": [
            {"text": s, "tokens": [it for it in items if it["i"] == i],
             "model": model[i], "rules": rules[i]}
            for i, s in enumerate(sents)
        ],  # fmt: skip
        "entities": names,
    }


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Türkçe merkezleme · model demosu</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1a;--mute:#6b6b66;--line:#e3e3de;--acc:#3b5bdb;--L:86%;--S:70%}
@media(prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecece8;--mute:#9a9a93;--line:#33332f;--acc:#8fa5ff;--L:28%;--S:45%}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:20px;margin:0 0 4px}p.sub{margin:0 0 16px;color:var(--mute);font-size:13px}
textarea{width:100%;min-height:110px;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink);font:inherit;resize:vertical}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0 22px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.go{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}button:disabled{opacity:.55;cursor:wait}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.n{color:var(--mute);font-size:12px;font-weight:600;letter-spacing:.04em}
.toks{margin:6px 0 12px;line-height:2.1}
.t{padding:2px 5px;border-radius:5px;margin-right:2px;white-space:nowrap}
.t.c{background:hsl(var(--h) var(--S) var(--L))}
.t.ph{border:1px dashed var(--mute);color:var(--mute);font-style:italic;font-size:13px}
.t.ph.c{color:var(--ink)}
.sys{display:grid;grid-template-columns:90px 1fr;gap:4px 12px;font-size:13.5px;padding-top:8px;border-top:1px solid var(--line)}
.sys+.sys{margin-top:8px}.who{color:var(--mute);font-weight:600}
.b{display:inline-block;padding:1px 9px;border-radius:99px;font-weight:600;font-size:12.5px;color:#fff;background:#868e96}
.b.Continue{background:#2f9e44}.b.Retain{background:#0c8599}.b.Smooth-Shift{background:#e67700}
.b.Rough-Shift{background:#d9480f}.b.NOCB{background:#868e96}.b.Establish{background:#4263eb}
.kv{color:var(--mute)}.kv b{color:var(--ink);font-weight:600}.cf span{margin-right:6px}
.cf i{color:var(--mute);font-style:normal;font-size:11.5px}
.diff{outline:2px solid #f08c00;outline-offset:2px}
.legend{font-size:13px;color:var(--mute);margin:-8px 0 16px}.legend .t{margin-right:6px}
.err{color:#e03131}
</style></head><body><main>
<h1>Türkçe merkezleme · model demosu</h1>
<p class="sub">Geçici, yerel arayüz. Renkli sözcükler modelin aynı varlık saydığı bahsetmeler.
Kesik çerçeveli sözcükler metinde yok: kuralların eklediği düşürülmüş özne / iyelik yer tutucuları.
Bir sözcüğün üzerine gelince DizgeBERT çözümlemesi görünür. Turuncu çerçeve: model ile kurallar farklı geçiş verdi.</p>
<textarea id="tx">Yaşlı bir oduncu ormanın kenarında yaşarmış. Her sabah baltasını alıp ormana gidermiş. Bir gün ormanda küçük bir kuş bulmuş. Kuşun kanadı kırıkmış. Oduncu onu evine götürmüş. Karısı kuşu görünce çok sevinmiş.</textarea>
<div class="row"><button class="go" id="go">Çözümle</button>
<button data-ex="Ali dün Ayşe'yi aradı. Ona kitabını geri verecekti. Ama evde yoktu. Annesi telefonu açtı.">Örnek: Ali ve Ayşe</button>
<button data-ex="Dün pazara gittim. Domates aldım. Çok pahalıydı. Satıcıya sordum. Bana indirim yapmadı.">Örnek: birinci kişi</button>
<span id="st" class="kv"></span></div>
<div id="legend" class="legend"></div><div id="out"></div>
<script>
const $=s=>document.querySelector(s), hue=c=>(parseInt(c.slice(1))*67+20)%360;
const el=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e};
function sysRow(name,u){const d=el('div','sys');d.append(el('div','who',name));const r=el('div');
  r.append(el('span','b '+u.transition,u.transition));
  const kv=el('span','kv');kv.innerHTML='  Cb: <b></b>  ·  Cp: <b></b>';const b=kv.querySelectorAll('b');
  b[0].textContent=u.cb||'—';b[1].textContent=u.cp||'—';r.append(kv);
  const cf=el('div','cf kv');cf.append('Cf: ');
  if(!u.cf.length)cf.append('—');
  u.cf.forEach((m,i)=>{const s=el('span');const w=el('b',null,m.label);if(/^C\d+$/.test(m.key)){w.className='t c';w.style.setProperty('--h',hue(m.key))}
    s.append(w);if(m.kind!=='noun')s.append(el('i',null,' '+({zero:'∅ özne',possessor:'∅ iyelik',pronoun:'zamir'}[m.kind]||m.kind)));
    if(i<u.cf.length-1)s.append(' ›');cf.append(s)});
  if(u.unresolved.length)cf.append(el('i',null,'  çözülemeyen: '+u.unresolved.join(', ')));
  r.append(cf);d.append(r);return d}
async function run(){const b=$('#go');b.disabled=true;$('#st').textContent='çözümleniyor…';$('#st').className='kv';
  try{const r=await fetch('/analyze',{method:'POST',body:JSON.stringify({text:$('#tx').value})});
    const d=await r.json();if(d.error)throw new Error(d.error);
    const out=$('#out');out.textContent='';const lg=$('#legend');lg.textContent='';
    const ents=Object.entries(d.entities);if(ents.length){lg.append('Modelin bulduğu varlıklar: ');
      ents.forEach(([c,n])=>{const t=el('span','t c',n);t.style.setProperty('--h',hue(c));lg.append(t)})}
    d.sentences.forEach((s,i)=>{const c=el('div','card');if(s.model.transition!==s.rules.transition)c.classList.add('diff');
      c.append(el('div','n','CÜMLE '+(i+1)));const tk=el('div','toks');
      s.tokens.forEach(t=>{const e=el('span','t'+(t.kind!=='tok'?' ph':'')+(t.cluster?' c':''),(t.kind!=='tok'?'∅ ':'')+t.form);
        if(t.cluster)e.style.setProperty('--h',hue(t.cluster));
        e.title=t.kind==='tok'?t.info:(t.kind==='zero'?'düşürülmüş özne (kural ekledi)':'düşürülmüş iyelik (kural ekledi)');tk.append(e,' ')});
      c.append(tk,sysRow('Model',s.model),sysRow('Kurallar',s.rules));out.append(c)});
    $('#st').textContent=d.sentences.length+' cümle'}
  catch(e){$('#st').textContent='Hata: '+e.message;$('#st').className='err'}b.disabled=false}
$('#go').onclick=run;document.querySelectorAll('[data-ex]').forEach(b=>b.onclick=()=>{$('#tx').value=b.dataset.ex;run()});
</script></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 200_000)
            text = json.loads(self.rfile.read(n) or b"{}").get("text", "")[:20_000]
            with LOCK:  # one model, one request at a time
                result = analyze(text)
        except Exception as e:  # shown in the page; this is a debugging tool
            result = {"error": f"{type(e).__name__}: {e}"}
        self._send(
            json.dumps(result, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def log_message(self, *args):
        pass


def main(argv):
    global PARSER, MODEL
    import logging

    import datasets
    from fastcoref import FCoref

    datasets.disable_progress_bar()
    logging.disable(logging.WARNING)  # fastcoref logs every prediction at INFO
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8765
    PARSER = JointParser()
    MODEL = FCoref(model_name_or_path=argv[0], nlp=None, enable_progress_bar=False)
    print(f"ready: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
