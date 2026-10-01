"""
Train a Turkish fastcoref (FCoref architecture) model on Turkish-ITCC chunks made by
tr_coref_data.py. The best checkpoint on the validation file (CoNLL F1) is saved to
OUT_DIR/model and loads with `fastcoref.FCoref(model_name_or_path=OUT_DIR/model)`.

Trained on ITCC (CC BY-NC-SA 4.0): the resulting model is non-commercial too.

Usage:
    python experiments/tr_coref_train.py FIT.jsonl VAL.jsonl OUT_DIR
        [--encoder dbmdz/bert-base-turkish-cased] [--epochs 15] [--max-tokens 1024]
"""

import sys
import types


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    fit, val, out = sys.argv[1:4]
    import spacy
    from fastcoref import CorefTrainer, TrainingArgs

    # ponytail: fastcoref's trainer imports wandb only to log; stub it rather than
    # install it. Installed after the imports above, so accelerate's availability
    # probe never sees the stub. Drop if fastcoref makes logging optional.
    stub = types.ModuleType("wandb")
    run = types.SimpleNamespace(summary={})
    stub.init = lambda **k: run
    stub.log = lambda *a, **k: None
    stub.run = run
    sys.modules["wandb"] = stub

    args = TrainingArgs(
        model_name_or_path=arg("--encoder", "dbmdz/bert-base-turkish-cased"),
        output_dir=out,
        overwrite_output_dir=True,
        epochs=float(arg("--epochs", 15)),
        max_tokens_in_batch=int(arg("--max-tokens", 1024)),
        logging_steps=50,
        eval_steps=int(arg("--eval-steps", 65)),
        learning_rate=float(arg("--lr", 1e-5)),
        head_learning_rate=float(arg("--head-lr", 3e-4)),
        cache_dir=arg("--cache", "cache"),
        # fastcoref's defaults (30, 0.4) keep almost no gold span early on with this
        # little data, and training collapses to "no antecedent anywhere"
        max_span_length=int(arg("--max-span", 8)),
        top_lambda=float(arg("--top-lambda", 1.0)),
    )
    trainer = CorefTrainer(
        args=args, train_file=fit, dev_file=val, nlp=spacy.blank("tr")
    )
    trainer.train()
    print("final:", trainer.evaluate(test=False, prefix="final"))


if __name__ == "__main__":
    main()
