"""
Throwaway local UI for the Turkish coreference model + centering.

Written for readers, not linguists: words that refer to the same person or thing share
an underline colour, dropped subjects / possessors appear in parentheses, and a thread
between consecutive sentences shows what ties them (or that the tie is cut). A dropped
subject says who the model tied it to: "(o = Ali)". "Show
details" adds the technical names (topic move, transition, Cb, Cp, Cf) and what the
rules say.
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

SPEAKERS = {"@1sg": "ben", "@2sg": "sen", "@1pl": "biz", "@2pl": "siz"}
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
        return {"sentences": [], "entities": []}
    stream, where = build_stream(parses)
    clusters = cluster_words(MODEL, stream) if MODEL is not None else {}

    # stream items, with the parse of real tokens for the tooltip
    items = [None] * len(stream)
    for (i, kind, pos), idx in where.items():
        item = {"i": i, "form": stream[idx], "kind": kind, "cluster": clusters.get(idx),
                "noun": False, "nom": False}  # fmt: skip
        if kind == "tok":
            t = parses[i][pos - 1]
            feats = " ".join(f"{k}={v}" for k, v in t.feats.items())
            item["info"] = f"{t.upos} · {t.deprel} → {t.head} · {feats}"
            item["noun"] = t.upos in ("NOUN", "PROPN")
            item["nom"] = t.feats.get("Case") == "Nom"
        items[idx] = item
    # a cluster is named after a noun in its plain form if there is one ("Ayşe", not
    # "Ayşe'yi"), a cluster of pronouns only after a placeholder ("ben", "o")
    names = {}
    for ok in (
        lambda it: it["noun"] and it["nom"],
        lambda it: it["noun"],
        lambda it: it["kind"] != "tok",
        lambda it: True,
    ):
        for it in items:
            if it["cluster"] and ok(it):
                plain = it["form"].replace("’", "'").split("'")[0]
                names.setdefault(it["cluster"], plain)
    order = list(dict.fromkeys(it["cluster"] for it in items if it["cluster"]))

    def report(rep):
        out = []
        for i, u in enumerate(rep.utterances):
            # an entity shows under its cluster name; one the model did not cluster
            # under the word as written (not lgram.tr's stem key: "annes", "telefo")
            lab = {}
            for m in u.cf:
                word = (
                    parses[i][m.pos - 1].form if m.kind == "noun" and m.pos else m.key
                )
                plain = word.replace("’", "'").split("'")[0]
                lab[m.key] = SPEAKERS.get(m.key) or names.get(m.key) or plain
            out.append(
                {
                    "transition": u.transition.value,
                    "move": u.topic_move,
                    "cb": lab.get(u.cb),
                    "cp": lab.get(u.cp),
                    "cb_key": u.cb,
                    "cp_key": u.cp,
                    "cf": [
                        {"key": m.key, "label": lab[m.key], "kind": m.kind}
                        for m in u.cf
                    ],
                    "unresolved": u.unresolved,
                }
            )
        return out

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
        # in order of first appearance: the page gives each one a thread colour
        "entities": [{"id": c, "name": names[c]} for c in order],
    }


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bu metin kimden bahsediyor?</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<style>
:root{
  --ground:#F2F4F3;--ink:#16202A;--soft:#5E6B75;--line:#D5DBDC;--field:#FFFFFF;
  --e0:#2F4B9A;--e1:#B23A48;--e2:#B57A08;--e3:#1B8282;--e4:#647A22;--e5:#7B3F8C;--e6:#B5561F;--e7:#2E6B4F;
  --ui:"Bricolage Grotesque",system-ui,"Segoe UI",sans-serif;--read:"Newsreader",Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#12171A;--ink:#E9EDEE;--soft:#93A0A8;--line:#2B343A;--field:#1A2126;
  --e0:#8FA6F2;--e1:#F08A95;--e2:#E8B24A;--e3:#5CCFCF;--e4:#B4CC6A;--e5:#D29AE3;--e6:#F0A070;--e7:#7FCFA6;
}}
*{box-sizing:border-box}
html{background:var(--ground)}
body{margin:0;color:var(--ink);font:400 16px/1.5 var(--ui)}
main{max-width:44rem;margin:0 auto;padding:clamp(28px,7vw,72px) 20px 96px}
h1{font:600 clamp(2rem,6vw,3.1rem)/1.04 var(--ui);letter-spacing:-.025em;margin:0 0 .55em;max-width:13ch}
.lede{color:var(--soft);margin:0 0 1.6rem;max-width:34rem}
textarea{display:block;width:100%;min-height:8.5rem;padding:16px 18px;border:1px solid var(--line);border-radius:4px;
  background:var(--field);color:var(--ink);font:400 1.2rem/1.5 var(--read);resize:vertical}
textarea:focus-visible,button:focus-visible,input:focus-visible{outline:2px solid var(--e0);outline-offset:3px}
.actions{display:flex;flex-wrap:wrap;align-items:baseline;gap:10px 20px;margin:14px 0 0}
#go{font:600 1rem var(--ui);padding:11px 20px;border:0;border-radius:4px;background:var(--ink);color:var(--ground);cursor:pointer}
#go:disabled{opacity:.6;cursor:progress}
.try{color:var(--soft);font-size:.92rem}
.try button{font:inherit;color:var(--ink);background:none;border:0;padding:0;margin-left:10px;cursor:pointer;
  text-decoration:underline;text-decoration-color:var(--line);text-underline-offset:4px;text-decoration-thickness:2px}
.try button:hover{text-decoration-color:var(--ink)}
#status{margin:18px 0 0;color:var(--soft)}
#status:empty{display:none}
#status.bad{color:var(--e1)}
#result[hidden]{display:none}
.who{margin:40px 0 6px;font-size:.95rem;color:var(--soft)}
.who span{margin-left:12px;white-space:nowrap}
.key{margin:0 0 30px;font-size:.9rem;color:var(--soft);max-width:36rem}
/* an entity: a thread stitched under the word */
.e{text-decoration:underline;text-decoration-color:var(--c);text-decoration-thickness:3px;text-underline-offset:5px;text-decoration-skip-ink:none}
.who .e{color:var(--ink);font-family:var(--read);font-size:1.1rem}
.g{color:var(--soft);font-style:italic}
.g.e{text-decoration-style:dotted}
.g b{font-style:normal;font-weight:500;color:var(--ink)}
/* the flow: sentences on a rail; the rail is the thread that ties them */
.flow{list-style:none;margin:0;padding:0}
.flow li{position:relative;padding-left:34px}
.sent{font:400 1.32rem/1.55 var(--read);padding:2px 0}
.sent::before{content:"";position:absolute;left:5px;top:.72em;width:11px;height:11px;border-radius:50%;background:var(--ink)}
.sent i.up,.sent i.down,.link::before{content:"";position:absolute;left:9px;width:3px;background:var(--c,var(--line))}
.sent i.up{top:0;height:.72em}
.sent i.down{top:calc(.72em + 11px);bottom:0}
.link{padding:9px 0 11px;font-size:.93rem;color:var(--soft)}
.link::before{top:0;bottom:0}
.link.cut::before,.sent i.cut{background:repeating-linear-gradient(to bottom,var(--line) 0 4px,transparent 4px 9px)}
.link b{font-weight:600;color:var(--ink)}
.link .e{color:var(--ink)}
.more{display:none;margin-top:5px;font-size:.84rem;line-height:1.55}
.more .differs{color:var(--e1)}
body.detail .more{display:block}
.toggle{display:flex;align-items:center;gap:9px;margin:34px 0 0;font-size:.92rem;color:var(--soft);cursor:pointer;width:fit-content}
.toggle input{width:17px;height:17px;accent-color:var(--ink);margin:0}
.fine{margin:14px 0 0;font-size:.84rem;color:var(--soft);max-width:36rem}
@media (prefers-reduced-motion:no-preference){
  .flow li{animation:rise .5s both;animation-delay:calc(var(--n)*70ms)}
  .link::before{animation:draw .45s both;animation-delay:calc(var(--n)*70ms);transform-origin:top}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}}
  @keyframes draw{from{transform:scaleY(0)}}
}
</style></head><body><main>
<h1>Bu metin kimden bahsediyor?</h1>
<p class="lede">Birkaç cümle yaz. Her cümlenin bir öncekine kimin ya da neyin üzerinden bağlandığını, bağın nerede koptuğunu gösterelim.</p>
<textarea id="tx" aria-label="Çözümlenecek metin" spellcheck="false">Ali dün Ayşe'yi aradı. Ona kitabını geri verecekti. Ama evde yoktu. Annesi telefonu açtı.</textarea>
<div class="actions"><button id="go">Bağları göster</button>
<span class="try">Örnek dene:
<button data-ex="Ali dün Ayşe'yi aradı. Ona kitabını geri verecekti. Ama evde yoktu. Annesi telefonu açtı.">Ali ile Ayşe</button>
<button data-ex="Dün pazara gittim. Domates aldım. Çok pahalıydı. Satıcıya sordum. Bana indirim yapmadı.">pazarda</button>
<button data-ex="Yaşlı bir oduncu ormanın kenarında yaşarmış. Her sabah baltasını alıp ormana gidermiş. Bir gün ormanda küçük bir kuş bulmuş. Kuşun kanadı kırıkmış. Oduncu onu evine götürmüş. Karısı kuşu görünce çok sevinmiş.">oduncu masalı</button>
</span></div>
<p id="status" role="status"></p>
<section id="result" hidden>
<p class="who" id="who"></p>
<p class="key">Altı aynı renkle çizili sözcükler aynı kişiyi ya da şeyi gösteriyor. Parantez içindekiler metinde yazmıyor: cümlenin söylenmeyen öznesi ya da sahibi; eşittirden sonra, modelin onu kime ya da neye bağladığı yazıyor. Cümleleri birleştiren ipin rengi, bağı kuranın rengi.</p>
<ol class="flow" id="flow"></ol>
<label class="toggle"><input type="checkbox" id="detail"> Ayrıntıları göster</label>
<p class="fine">Deneme sürümü. Model bağları her zaman doğru kurmuyor; ayrıntılarda kural tabanlı yöntemin ne dediğini de görebilirsin.</p>
</section>
<script>
const $=s=>document.querySelector(s);
const el=(tag,cls,txt)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e};
const PLAIN={Continue:'Aynı konu sürüyor','Retain':'Konu aynı, odak kayıyor','Smooth-Shift':'Konu değişti','Rough-Shift':'Konu sertçe değişti',NOCB:'Bağ koptu'};
const MOVE={'devam':'Aynı konu sürüyor','yumuşak dönüş':'Konu, az önce anılan bir başkasına geçti','içerme':'Yeni konu, öncekine ait ya da onun parçası','tam dönüş':'Yepyeni bir konu'};
const KIND={zero:'söylenmeyen özne',possessor:'söylenmeyen sahip',pronoun:'zamir'};
let color=new Map(),named=new Map();
const paint=(node,key)=>{if(color.has(key)){node.classList.add('e');node.style.setProperty('--c','var(--e'+color.get(key)+')')}return node};
function words(tokens){const p=el('span');let open=true;
  tokens.forEach(t=>{const ghost=t.kind!=='tok',close=/^[.,!?;:…)\]’”]+$/.test(t.form);
    if(!open&&!close)p.append(' ');
    // a dropped subject / owner names who the model tied it to: "(o = Ali)"
    const ref=ghost&&named.get(t.cluster),tied=ref&&ref.toLowerCase()!==t.form.toLowerCase();
    const w=el('span',ghost?'g':null,ghost?'('+t.form+(tied?' = ':')'):t.form);if(tied)w.append(el('b',null,ref),')');
    if(t.cluster)paint(w,t.cluster);
    w.title=ghost?'Metinde yazmıyor: '+KIND[t.kind]+(tied?'. Modele göre: '+ref:t.cluster?'':'. Model bunu kimseye bağlamadı'):t.info;
    p.append(w);open=/^[(\[‘“]+$/.test(t.form)});
  return p}
function tech(name,u,other){const d=el('div');
  d.append(name+': '+(u.move?u.move+' ('+u.transition+')':u.transition)+', bağ (Cb) '+(u.cb||'yok')+', odak (Cp) '+(u.cp||'yok')+'. Sıra (Cf): '+(u.cf.map(m=>m.label).join(', ')||'boş')+'.');
  if(other&&(other.transition!==u.transition||other.move!==u.move))d.className='differs';return d}
function render(d){color=new Map(d.entities.map((e,i)=>[e.id,i%8]));named=new Map(d.entities.map(e=>[e.id,e.name]));
  const who=$('#who');who.textContent=d.entities.length?'Metindeki kişi ve şeyler:':'Model bu metinde tekrar eden bir kişi ya da şey bulamadı.';
  d.entities.forEach(e=>who.append(paint(el('span',null,e.name),e.id)));
  const flow=$('#flow');flow.textContent='';const S=d.sentences;
  const tie=u=>!u||u.transition==='NOCB'?null:u.cb_key;
  const thread=(node,u)=>{if(tie(u))node.style.setProperty('--c',color.has(u.cb_key)?'var(--e'+color.get(u.cb_key)+')':'var(--soft)');else node.classList.add('cut')};
  S.forEach((s,i)=>{const u=s.model;
    if(i){const li=el('li','link');li.style.setProperty('--n',i*2-1);thread(li,u);
      li.append(el('b',null,MOVE[u.move]||PLAIN[u.transition]));
      if(u.cb){li.append('. Bağ: ');li.append(paint(el('span',null,u.cb),u.cb_key));
        if(u.cp&&u.cp_key!==u.cb_key){li.append(', odak: ');li.append(paint(el('span',null,u.cp),u.cp_key))}li.append('.')}
      else li.append(u.move==='içerme'?'. Önceki cümlede anılanın bir üyesi ya da parçası.':'. Önceki cümleyle ortak bir kişi ya da şey yok.');
      const m=el('div','more');m.append(tech('Model',u,null));if(s.rules)m.append(tech('Kurallar',s.rules,u));li.append(m);flow.append(li)}
    const li=el('li','sent');li.style.setProperty('--n',i*2);
    if(i){const up=el('i','up');thread(up,u);li.append(up)}
    if(i<S.length-1){const dn=el('i','down');thread(dn,S[i+1].model);li.append(dn)}
    li.append(words(s.tokens));flow.append(li)});
  $('#result').hidden=false}
async function run(){const b=$('#go'),st=$('#status');b.disabled=true;b.textContent='Okuyor…';st.className='';st.textContent='';
  try{const r=await fetch('/analyze',{method:'POST',body:JSON.stringify({text:$('#tx').value})});
    const d=await r.json();if(d.error)throw new Error(d.error);
    if(!d.sentences.length){$('#result').hidden=true;st.textContent='Çözümlenecek bir cümle bulunamadı. Birkaç cümle yaz ya da bir örnek seç.'}
    else{render(d);if(d.sentences.length<2)st.textContent='Bağ görmek için en az iki cümle gerekiyor.'}}
  catch(e){$('#result').hidden=true;st.className='bad';st.textContent='Çözümleme yapılamadı: '+e.message+'. Sunucu hâlâ çalışıyor mu?'}
  b.disabled=false;b.textContent='Bağları göster'}
$('#go').onclick=run;
document.querySelectorAll('[data-ex]').forEach(b=>b.onclick=()=>{$('#tx').value=b.dataset.ex;run()});
const box=$('#detail');try{box.checked=localStorage.getItem('detail')==='1'}catch(e){}
const sync=()=>{document.body.classList.toggle('detail',box.checked);try{localStorage.setItem('detail',box.checked?'1':'0')}catch(e){}};
box.onchange=sync;sync();
const q=new URLSearchParams(location.search);if(q.has('ayrinti')){box.checked=true;sync()}if(q.has('ornek'))run();
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
