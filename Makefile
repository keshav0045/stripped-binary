# End-to-end pipeline. Stages 1a-1b run inside the pipeline container
# (`make shell`); everything after runs anywhere with requirements installed.
PY ?= python
IMAGE = fnname-pipeline
JOBS ?= 8

image:
	docker build --platform linux/amd64 -f docker/Dockerfile.pipeline -t $(IMAGE) .

shell:
	docker run --rm -it --platform linux/amd64 -v $(PWD):/app $(IMAGE) bash

build:        ## 1a: compile every project x config, keep unstripped + stripped
	$(PY) -m fnname.build --projects configs/projects.yaml --work work

decompile:    ## 1b: Ghidra headless over all stripped binaries
	$(PY) -m fnname.decompile --work work --jobs $(JOBS)

dataset:      ## 1c+1d: pair with ground truth, split by project, dedupe
	$(PY) -m fnname.pair --work work --out data/functions.jsonl
	$(PY) -m fnname.split --data data/functions.jsonl --out data

baseline:     ## 2: frequency + TF-IDF kNN
	$(PY) -m fnname.baselines.knn --train data/train.jsonl --test data/test.jsonl --out preds/knn.jsonl --save models/knn.pkl

lora:         ## 3: fine-tune (GPU) and predict
	$(PY) -m fnname.llm.train_lora --train data/train.jsonl --val data/val.jsonl --out models/lora
	$(PY) -m fnname.llm.predict_lora --adapter models/lora --test data/test.jsonl --out preds/lora.jsonl

zeroshot:     ## 3: zero-shot API baseline on a 2K sample
	$(PY) -m fnname.llm.zero_shot --test data/test.jsonl --out preds/zeroshot.jsonl --sample 2000

eval:         ## 4: results tables
	$(PY) -m fnname.evaluate --test data/test.jsonl --failures 15 preds/*.jsonl

serve:        ## 5
	FNNAME_MODEL=models/knn.pkl uvicorn serve.app:app --port 8000

test:
	$(PY) -m pytest -q tests

.PHONY: image shell build decompile dataset baseline lora zeroshot eval serve test
