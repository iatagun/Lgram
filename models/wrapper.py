# wrapper.py

import runpy
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# —— 1) spaCy GPU bindirmesini dene, başarısızsa devam et
try:
    spacy.require_gpu()
    print("spaCy GPU’da çalışıyor ✅")
except Exception as e:
    print(f"spaCy GPU’a geçemedi ({e}), CPU’da devam ediliyor ⚠️")

# —— 2) Model.from_pretrained’i yakalayarak GPU’ya taşı
_orig_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained
def _gpu_from_pretrained(*args, **kwargs):
    model = _orig_from_pretrained(*args, **kwargs)
    if torch.cuda.is_available():
        model.to("cuda")
        print("Transformer modeli GPU’ya taşındı 🚀")
    else:
        print("CUDA bulunamadı, model CPU’da kalacak")
    return model
AutoModelForSeq2SeqLM.from_pretrained = _gpu_from_pretrained

# —— 3) Tokenizer() çıktılarındaki tensor’ları da GPU’ya gönder
_orig_tokenize = AutoTokenizer.__call__
def _gpu_tokenize(self, *args, **kwargs):
    toks = _orig_tokenize(self, *args, **kwargs)
    if torch.cuda.is_available():
        for k, v in toks.items():
            if hasattr(v, "to"):
                toks[k] = v.to("cuda")
    return toks
AutoTokenizer.__call__ = _gpu_tokenize

# —— 4) Orijinal chunk.py’yi hiç değiştirmeden çalıştır
runpy.run_path("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\chunk.py", run_name="__main__")
