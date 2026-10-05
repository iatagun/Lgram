"""
Hand-labelling tool for a Turkish coreference / centering test set that no model has seen.

Every result so far rests on Turkish-ITCC, which the teacher model was trained on. This
makes an independent yardstick: short windows of consecutive sentences, with the mention
slots lgram.tr finds. The page asks one question at a time, for every dropped subject,
implicit possessor and pronoun: who or what is it? The annotator answers by clicking the
word in the text (several words for a plural), or "not in the text", or "no such element".
A noun that looks like an earlier one ("tavşana ... tavşanı") is asked about too: the same
thing again, or a new one? Each sentence ends with: who or what is it about (the centre of
attention)? A last step ties different nouns that name the same thing ("Ali ...
adamcağız"). An earlier free-form design (chains, relations, added elements) was too much
to hold in mind. Labelling is blind: no model prediction is shown.

Labels: {window id: {"answers": {slot id: {"to": [ids]} | {"none": true} | {"bogus": true}},
"links": [[token id, token id], ...], "done": bool, "note": str}}; slot id =
"sentence:kind:pos" with kind zero / possessor / pronoun / noun, or "sentence:focus:0";
an id in "to" is a slot id or a token id "sentence:tok:id". "none" on a noun = a new
referent, on a focus = the sentence is about nothing in particular.

    python experiments/tr_label.py prep BATCH.json NAME=TEXT.txt [NAME=TEXT.txt ...] [--per 20]
    python experiments/tr_label.py serve BATCH.json [--port 8766]

prep: TEXT.txt holds documents separated by blank lines; NAME is the genre label. It
needs the [tr] extra (parser). serve needs nothing but the standard library and writes
every change to BATCH.labels.json next to BATCH.json.
"""

import html
import json
import os
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_DEICTIC = {"ben", "sen", "biz", "siz", "benim", "senin", "bizim", "sizin"}
_CONTINUES = {"ancak", "fakat", "ama", "oysa", "çünkü", "ayrıca", "sonra", "böylece", "yine",
              "üstelik", "ise", "ve", "de", "da", "bunun", "buna", "bundan", "daha", "söz"}  # fmt: skip
_POINTS = {"bu", "şu", "o", "bunlar", "şunlar", "onlar", "böyle", "öyle", "şöyle"}


def _opens(tokens, analyze_parsed):
    """True if the sentence names something and leans on no earlier sentence: no
    third-person pronoun, dropped subject or implicit possessor ("ben / sen" point at
    the speakers, not at the text, so they are fine)."""
    # "Fakat ...", "Bu olay ..." carry on from something said before
    if tokens[0].form.lower() in _CONTINUES or any(t.form.lower() in _POINTS for t in tokens):
        return False
    found = []
    analyze_parsed(["s"], [tokens], identity=lambda i, m: found.append(m))
    for m in found:
        if m.kind in ("zero", "possessor") and m.form not in _DEICTIC:
            return False
        if m.kind == "pronoun" and tokens[m.pos - 1].feats.get("Person") not in ("1", "2"):
            return False
    return any(m.kind == "noun" for m in found)


def prep(out, sources, per=20, seed=0, max_sents=6):
    from lgram.tr.centering import analyze_parsed
    from lgram.tr.parser import JointParser, split_sentences

    parser = JointParser()
    rng = random.Random(seed)
    windows = []
    for genre, path in sources:
        blocks = [" ".join(html.unescape(b).split())
                  for b in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8"))]  # fmt: skip
        picks = list(range(len(blocks)))
        rng.shuffle(picks)
        kept = 0
        for b in picks:
            if kept >= per:
                break
            # folk-tale files open a tale with its title in capitals
            block = re.sub(r"^(?:[A-ZÇĞİÖŞÜ’'\-]{2,}\s+)+", "", blocks[b])
            sents = split_sentences(block)
            if len(sents) < 4:
                continue
            parses = [parser.parse(s) for s in sents]
            # a window must open on a sentence that stands on its own: one that starts
            # with "(o) ... annesiyle (onun)" has its antecedents outside the window
            starts = [i for i in range(len(sents) - 3) if _opens(parses[i], analyze_parsed)]
            if not starts:
                continue
            start = rng.choice(starts)
            sents, parses = sents[start : start + max_sents], parses[start : start + max_sents]
            # a 2-word line or a 45-word list is a splitter miss, not a sentence to label
            if any(not 3 <= len(p) <= 45 for p in parses):
                continue
            # so is a table row of figures, or a sentence cut at "II."
            if any(sum(ch.isdigit() for ch in x) > 0.12 * len(x.replace(" ", "")) for x in sents) or any(
                re.search(r"\b[IVX]+\s*\.$", x) for x in sents
            ):
                continue
            slots = set()
            analyze_parsed(sents, parses,
                           identity=lambda i, m: slots.add((i, m.kind, m.pos, m.form)))  # fmt: skip
            windows.append({
                "id": f"{genre}-{kept + 1:02d}",
                "genre": genre,
                # the whole block and its neighbours, for the annotator to read around
                # the window (neighbours may belong to another text in the same file)
                "context": {"before": blocks[max(0, b - 2) : b], "block": block,
                            "after": blocks[b + 1 : b + 3]},  # fmt: skip
                "sentences": [
                    {"text": s,
                     "tokens": [{"id": t.id, "form": t.form, "upos": t.upos, "feats": t.feats,
                                 "head": t.head, "deprel": t.deprel} for t in p],
                     "slots": sorted([{"kind": k, "pos": pos, "form": form}
                                      for j, k, pos, form in slots if j == i],
                                     key=lambda x: (x["pos"], x["kind"]))}  # fmt: skip
                    for i, (s, p) in enumerate(zip(sents, parses))
                ],
            })  # fmt: skip
            kept += 1
            print(f"{genre} {kept}/{per}", flush=True)
    Path(out).write_text(json.dumps(windows, ensure_ascii=False, indent=1), encoding="utf-8")  # fmt: skip
    n = sum(len(w["sentences"]) - 1 for w in windows)
    print(f"{len(windows)} windows, {n} transitions -> {out}")


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gönderim etiketleme</title>
<style>
:root{--ground:#F2F4F3;--ink:#16202A;--soft:#5E6B75;--line:#D5DBDC;--field:#FFFFFF;--bad:#B23A48;--hi:#F6D365;--ok:#1B8282}
@media (prefers-color-scheme:dark){:root{--ground:#12171A;--ink:#E9EDEE;--soft:#93A0A8;--line:#2B343A;--field:#1A2126;--bad:#F08A95;--hi:#8A6A12;--ok:#5CCFCF}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:400 16px/1.5 system-ui,"Segoe UI",sans-serif}
main{max-width:50rem;margin:0 auto;padding:18px 20px 80px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin-bottom:16px;color:var(--soft)}
header b{color:var(--ink);font-weight:600}
.grow{flex:1}
button{font:inherit;color:var(--ink);background:var(--field);border:1px solid var(--line);border-radius:6px;padding:8px 14px;cursor:pointer}
button:hover{border-color:var(--ink)} button.on{background:var(--ink);color:var(--ground);border-color:var(--ink)}
button.small{padding:4px 10px;font-size:.88rem}
button:focus-visible,textarea:focus-visible,summary:focus-visible{outline:2px solid var(--ok);outline-offset:2px}
#text{font:400 1.3rem/2.2 Georgia,"Times New Roman",serif;margin:0 0 18px;padding:0;list-style:none}
#text li{display:flex;gap:12px;border-top:1px solid var(--line);padding:4px 0}
#text li.now{background:color-mix(in srgb,var(--hi) 16%,transparent)}
#text li>.n{flex:0 0 1.2rem;color:var(--soft);font:400 .8rem/3.6 system-ui,sans-serif;text-align:right}
.t{border-radius:3px;padding:1px 2px}
.pick .t.can{cursor:pointer} .pick .t.can:hover{background:color-mix(in srgb,var(--ok) 25%,transparent)}
.g{font-style:italic;color:var(--soft)}
.res{font:600 .8rem/1 system-ui,sans-serif;font-style:normal;color:var(--ok);margin-left:2px}
.cur{background:var(--hi);color:#16202A;font-style:normal;font-weight:600}
.anchor{box-shadow:inset 0 -3px 0 var(--hi)}
.chosen{background:color-mix(in srgb,var(--ok) 40%,transparent)}
.bogus{text-decoration:line-through;opacity:.5}
.t sup{font:600 .62rem/1 system-ui,sans-serif;color:var(--ok);margin-left:1px}
#ask{position:sticky;bottom:0;background:var(--field);border:1px solid var(--line);border-radius:8px;padding:14px 16px;box-shadow:0 -6px 18px color-mix(in srgb,var(--ink) 8%,transparent)}
#q{font-size:1.15rem;margin:0 0 4px} #q b{font-weight:600}
#how{color:var(--soft);margin:0 0 12px;font-size:.92rem}
#btns{display:flex;flex-wrap:wrap;gap:8px}
kbd{font:600 .78rem ui-monospace,Consolas,monospace;border:1px solid var(--line);border-radius:3px;padding:0 4px;margin-left:4px;color:var(--soft)}
details{margin:0 0 14px;color:var(--soft)} summary{cursor:pointer}
#ctx p{margin:.45em 0;font:400 1rem/1.6 Georgia,serif;color:var(--ink)} #ctx .dim{color:var(--soft)}
mark{background:color-mix(in srgb,var(--hi) 45%,transparent);color:inherit}
textarea{width:100%;min-height:2.6rem;margin:14px 0 0;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--field);color:var(--ink);font:inherit}
#saved.bad{color:var(--bad);font-weight:600}
</style></head><body><main>
<header><button class="small" id="prev">←</button><span><b id="prog"></b></span><button class="small" id="next">→</button><span class="grow"></span><span id="saved"></span></header>
<details><summary>Nasıl yapılır</summary>
<p>Sarıyla işaretli ifade için alttaki soruyu yanıtla: yanıtın metinde geçtiği sözcüğe tıkla. Bir kişi ya da şey birkaç kez anılıyorsa hangisine tıkladığın önemli değil.</p>
<p><b>Var ama metinde geçmiyor</b> (<kbd>M</kbd>): bir özne / tamlayan var, ama bu tümcelerde adı geçmiyor. <b>Yok</b> (<kbd>Y</kbd>): burada öyle bir öğe hiç yok: kalıp söz ("Gel zaman git zaman"), ya da özne zaten yazılı. <b>Birden fazla:</b> ifade birkaç kişiyi ya da şeyi birlikte gösteriyor ("ayı ile tilki … paylaşırlarmış"): hepsine tıkla, sonra Tamam.</p>
<p><b>Yinelenen ad:</b> bir ad daha önce geçen bir ada benziyorsa sorulur ("tavşana … tavşanı"): aynı şeyse öncekine tıkla, değilse <kbd>H</kbd>.</p>
<p><b>Dikkat odağı</b> (◎): her tümcenin sonunda sorulur: tümce asıl kimin / neyin hakkında? Yazılmamış özneyse parantezli ifadeye tıkla. Tümce yalnızca bir durum bildiriyorsa ("Aradan uzun zaman geçmiş") <kbd>M</kbd>.</p>
<p><b>Geri al</b> (<kbd>Z</kbd>): son tıklamanı geri alır; art arda basarak daha eskilere dönersin.</p>
<p>Sorular bitince: aynı kişiyi ya da şeyi başka bir adla gösteren sözcükler varsa ("Ali … adamcağız") birine, sonra ötekine tıkla. Yoksa doğrudan Enter.</p></details>
<details id="ctxbox"><summary>Metnin tamamı</summary><div id="ctx"></div></details>
<ol id="text"></ol>
<div id="ask"><p id="q"></p><p id="how"></p><div id="btns"></div></div>
<textarea id="note" placeholder="Not (isteğe bağlı)" aria-label="Not"></textarea>
<script>
const $=s=>document.querySelector(s),el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
let W=[],L={},cur=0,q=0,multi=false,picks=[],sel=null,timer=null,hist=[];
const snap=()=>hist.push({l:JSON.stringify(lab()),q});
const lab=()=>L[W[cur].id]||(L[W[cur].id]={answers:{},links:[],done:false,note:''});
const tid=(i,t)=>i+':tok:'+t;
const plain=f=>f.replace('’',"'").split("'")[0];
// ponytail: a shared 4-letter start stands in for "same noun again"; it over-asks ("at ... ateş",
// answered with one key) and misses synonyms, which the last step still ties by hand
const same=(a,b)=>{let n=0;while(n<a.length&&n<b.length&&a[n]===b[n])n++;return n>=Math.max(2,Math.min(4,a.length,b.length))};
function items(){ // everything on screen, in reading order; questions are the items with .ask
  const out=[],seen=[];
  W[cur].sentences.forEach((S,i)=>{const root=(S.tokens.find(t=>t.head===0)||S.tokens[S.tokens.length-1]).id,after={},pron={},noun={};
    S.slots.forEach(s=>{const id=i+':'+s.kind+':'+s.pos;
      if(s.kind==='pronoun'){if((S.tokens[s.pos-1].feats||{}).PronType!=='Int')pron[s.pos]=id} // "ne, kim" ask, they do not refer
      else if(s.kind==='noun')noun[s.pos]=id;
      else(after[s.pos||root]=after[s.pos||root]||[]).push({id,i,form:s.form,ghost:true,ask:s.kind,anchor:s.pos||root})});
    S.tokens.forEach(t=>{let ask=pron[t.id]?'pronoun':null,qid=pron[t.id];
      if(noun[t.id]){const k=plain(t.form).toLocaleLowerCase('tr');if(seen.some(o=>same(o,k))){ask='noun';qid=noun[t.id]}seen.push(k)}
      out.push({id:tid(i,t.id),i,form:t.form,tok:t,ask,qid});
      (after[t.id]||[]).sort((x,y)=>(x.ask!=='possessor')-(y.ask!=='possessor')).forEach(g=>out.push({...g,qid:g.id,of:t.form}))});
    // asked last in its sentence, once the dropped elements have their answers
    out.push({id:i+':focus:0',qid:i+':focus:0',i,form:'◎',ask:'focus'})});
  return out}
function name(id,its,depth){const it=its.find(x=>x.id===id||x.qid===id);if(!it)return'?';
  const a=it.qid&&lab().answers[it.qid];
  if(a&&a.to&&depth<8)return a.to.map(t=>name(t,its,depth+1)).join(' + ');
  return it.ghost?it.form:plain(it.form)}
function groups(its){ // nouns tied in the last step: token id -> number
  const p={},find=x=>p[x]===undefined||p[x]===x?x:(p[x]=find(p[x]));
  lab().links.forEach(([a,b])=>{p[find(a)]=find(b)});
  const num={},out={};its.forEach(it=>{if(p[it.id]===undefined&&!lab().links.some(l=>l.includes(it.id)))return;const r=find(it.id);out[it.id]=num[r]||(num[r]=Object.keys(num).length+1)});
  return out}
function draw(){const w=W[cur],a=lab(),its=items(),Q=its.filter(x=>x.ask),cq=Q[q],review=q>=Q.length,num=groups(its);
  const done=W.filter(v=>(L[v.id]||{}).done).length;
  $('#prog').textContent='Metin '+(cur+1)+' / '+W.length+(a.done?' ✓':'');
  const ol=$('#text');ol.textContent='';ol.className='pick';
  w.sentences.forEach((S,i)=>{const li=el('li',cq&&cq.ask==='focus'&&cq.i===i?'now':null),p=el('span');li.append(el('span','n',String(i+1)),p);let open=true;
    its.filter(x=>x.i===i).forEach(it=>{const close=/^[.,!?;:…)\]’”]+$/.test(it.form);if(!open&&!close)p.append(' ');
      const ans=it.qid&&a.answers[it.qid],soft=it.ghost||it.ask==='focus',e=el('span','t'+(soft?' g':''),it.ghost?'('+it.form+')':it.form);
      if(ans){if(ans.bogus)e.classList.add('bogus');
        else if(!ans.none)e.append(el('span','res',' = '+name(it.qid,its,0)));
        else if(it.ask!=='noun')e.append(el('span','res',it.ask==='focus'?' = odak yok':' = metinde yok'))}
      if(num[it.id])e.append(el('sup',null,String(num[it.id])));
      const mine=cq&&it.qid===cq.qid,pid=it.ask==='noun'?it.id:it.qid||it.id; // a noun is clicked as the word it is
      if(mine)e.classList.add('cur');
      if(cq&&cq.ghost&&it.tok&&it.i===cq.i&&it.tok.id===cq.anchor)e.classList.add('anchor');
      if(picks.includes(pid)||sel===it.id)e.classList.add('chosen');
      const word=/\p{L}|\p{N}/u.test(it.form);
      if(!review&&!mine&&word){e.classList.add('can');e.onclick=()=>choose(pid)}
      else if(review&&it.ask&&!(sel&&it.ask==='noun')){e.classList.add('can');e.title='Bu soruyu yeniden yanıtla';e.onclick=()=>{q=Q.indexOf(it);sel=null;draw()}}
      else if(review&&word){e.classList.add('can');e.onclick=()=>pair(it.id)}
      p.append(e);open=/^[(\[‘“]+$/.test(it.form)});
    ol.append(li)});
  const C=w.context,cx=$('#ctx');cx.textContent='';$('#ctxbox').hidden=!C;
  if(C){C.before.forEach(b=>cx.append(el('p','dim',b)));const p=el('p'),last=w.sentences[w.sentences.length-1].text;
    const a0=C.block.indexOf(w.sentences[0].text),a1=C.block.indexOf(last,Math.max(a0,0));
    if(a0>=0&&a1>=0)p.append(C.block.slice(0,a0),el('mark',null,C.block.slice(a0,a1+last.length)),C.block.slice(a1+last.length));else p.textContent=C.block;
    cx.append(p);C.after.forEach(b=>cx.append(el('p','dim',b)))}
  const qe=$('#q'),how=$('#how'),bt=$('#btns');qe.textContent='';bt.textContent='';
  const btn=(txt,key,fn,on)=>{const b=el('button',on?'on':null,txt);if(key)b.append(el('kbd',null,key));b.onclick=fn;bt.append(b)};
  const undoBtn=()=>{if(hist.length||q>0)btn('↶ Geri al','Z',back)};if(!review)undoBtn();
  if(review){qe.append(el('b',null,'Sorular bitti. '),'Aynı kişiyi ya da şeyi başka bir adla gösteren sözcükler var mı? ("Ali … adamcağız")');
    how.textContent=sel?'Şimdi aynı kişiyi ya da şeyi gösteren öteki ada tıkla (aynı ikiliye yeniden tıklarsan bağ kalkar).':'Varsa birine, sonra ötekine tıkla. Yoksa Enter. Bir yanıtı değiştirmek için yanıtlı ifadeye tıkla.';
    undoBtn();btn(cur<W.length-1?'Bitti, sonraki metin':'Bitti','Enter',finish,true)}
  else{const k=cq.ask,who={zero:['«'+cq.of+'»',': kim / ne?',' (yazılmamış özne)'],possessor:['«'+cq.of+'»',': kimin / neyin?',' (yazılmamış tamlayan)'],
      pronoun:['«'+cq.form+'»',': kimi / neyi gösteriyor?',' (adıl)'],noun:['«'+cq.form+'»',': daha önce geçen bir kişiyi ya da şeyi mi gösteriyor?',' (yinelenen ad)'],
      focus:[(cq.i+1)+'. tümce',': kimin / neyin hakkında?',' (dikkat odağı)']}[k];
    // [M: it exists but is not in the text, Y: there is no such element here]
    const no={zero:['Öznesi var ama metinde geçmiyor','Yazılmamış özne yok (kalıp söz ya da özne zaten yazılı)'],possessor:['Tamlayanı var ama metinde geçmiyor','Tamlayanı yok'],
      pronoun:['Gösterdiği şey metinde geçmiyor','Hiçbir şeyi göstermiyor'],noun:['Hayır, yeni bir kişi ya da şey'],focus:['Belli bir odağı yok']}[k];
    qe.append('Soru '+(q+1)+' / '+Q.length+' — ',el('b',null,who[0]),who[1],el('span',null,who[2]));qe.lastChild.style.color='var(--soft)';
    how.textContent=multi?'Hepsine tıkla, sonra Tamam.':k==='noun'?'Gösteriyorsa o sözcüğe tıkla.':k==='focus'?'Tümcenin asıl sözünü ettiği kişiye ya da şeye tıkla (parantezli ifade de olur).':'Yanıtın geçtiği sözcüğe metinde tıkla.';
    if(multi)btn('Tamam ('+picks.length+')','Enter',()=>picks.length&&answer({to:picks}),true);
    btn(no[0],k==='noun'?'H':'M',()=>answer({none:true}));if(no[1])btn(no[1],'Y',()=>answer({bogus:true}));
    btn('Birden fazla','B',()=>{multi=!multi;picks=[];draw()},multi)}
  $('#note').value=a.note||''}
function choose(id){if(multi){picks=picks.includes(id)?picks.filter(x=>x!==id):[...picks,id];draw()}else answer({to:[id]})}
function answer(v){const Q=items().filter(x=>x.ask);snap();lab().answers[Q[q].qid]=v;lab().done=false;
  do q++;while(q<Q.length&&lab().answers[Q[q].qid]); // on to the next open question, not through answered ones
  multi=false;picks=[];save();draw()}
function back(){ // undo the last click of any kind; after a reload there is no history, so step back one question
  multi=false;picks=[];sel=null;
  if(hist.length){const h=hist.pop(),note=lab().note;L[W[cur].id]={...JSON.parse(h.l),note,done:false};q=h.q}
  else if(q>0){q--;const Q=items().filter(x=>x.ask);delete lab().answers[Q[q].qid];lab().done=false}
  else return;
  save();draw()}
function pair(id){const a=lab();if(!sel){sel=id;return draw()}if(sel===id){sel=null;return draw()}
  snap();const k=a.links.findIndex(l=>l.includes(sel)&&l.includes(id));if(k>=0)a.links.splice(k,1);else a.links.push([sel,id]);sel=null;a.done=false;save();draw()}
function finish(){lab().done=true;save();if(cur<W.length-1)go(cur+1);else draw()}
function save(){clearTimeout(timer);const id=W[cur].id,body=JSON.stringify({id,label:L[id]});
  timer=setTimeout(async()=>{const s=$('#saved');try{const r=await fetch('/save',{method:'POST',body});if(!r.ok)throw 0;s.className='';s.textContent='kaydedildi · biten metin: '+W.filter(v=>(L[v.id]||{}).done).length}
    catch(e){s.className='bad';s.textContent='KAYDEDİLEMEDİ: sunucu çalışıyor mu?'}},200)}
function go(n){cur=Math.max(0,Math.min(W.length-1,n));multi=false;picks=[];sel=null;hist=[];
  const Q=items().filter(x=>x.ask),k=Q.findIndex(x=>!lab().answers[x.qid]);q=k<0?Q.length:k;
  if(k>=0)lab().done=false; // finished before a kind of question was added
  draw();scrollTo(0,0)}
$('#prev').onclick=()=>go(cur-1);$('#next').onclick=()=>go(cur+1);
$('#note').oninput=e=>{lab().note=e.target.value;save()};
document.onkeydown=e=>{if(e.target.id==='note'||e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key.toLowerCase(),Q=items().filter(x=>x.ask),review=q>=Q.length;
  if(k==='backspace'||k==='z'){e.preventDefault();back()}
  else if(k==='enter'){e.preventDefault();if(review)finish();else if(multi&&picks.length)answer({to:picks})}
  else if(k==='escape'){multi=false;picks=[];sel=null;draw()}
  else if(review)return;
  else if(k==='m'||k==='h')answer({none:true});else if(k==='y'&&Q[q].ask!=='noun'&&Q[q].ask!=='focus')answer({bogus:true});else if(k==='b'){multi=!multi;picks=[];draw()}};
fetch('/data').then(r=>r.json()).then(d=>{W=d.windows;L=d.labels;
  const k=W.findIndex((w,n)=>{cur=n;const l=L[w.id];return !l||!l.done||items().some(x=>x.ask&&!l.answers[x.qid])});go(k<0?0:k)});
</script></main></body></html>"""


def serve(batch, port=8766):
    batch = Path(batch)
    out = batch.with_suffix(".labels.json")
    windows = json.loads(batch.read_text(encoding="utf-8"))
    labels = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/data":
                body = json.dumps({"windows": windows, "labels": labels}, ensure_ascii=False)  # fmt: skip
                self._send(body.encode("utf-8"), "application/json; charset=utf-8")
            else:
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):
            n = min(int(self.headers.get("Content-Length", 0)), 1_000_000)
            try:
                msg = json.loads(self.rfile.read(n))
                with lock:  # write to a temp file first: a crash must not eat the labels
                    labels[msg["id"]] = msg["label"]
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")  # fmt: skip
                    os.replace(tmp, out)
                self._send(b"{}", "application/json")
            except Exception as e:  # the page shows "not saved"
                self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8", 500)

        def log_message(self, *args):
            pass

    print(f"ready: http://127.0.0.1:{port}  (labels -> {out})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    arg = lambda name, d: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d  # noqa: E731
    if len(sys.argv) >= 3 and sys.argv[1] == "prep":
        srcs = [tuple(a.split("=", 1)) for a in sys.argv[3:] if "=" in a]
        prep(sys.argv[2], srcs, per=arg("--per", 20))
    elif len(sys.argv) >= 3 and sys.argv[1] == "serve":
        serve(sys.argv[2], arg("--port", 8766))
    else:
        sys.exit(__doc__)
