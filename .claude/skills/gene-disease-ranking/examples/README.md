# Example inputs

A four-disease smoke test. `candidates.txt` is deliberately packed with sibling
genes (GPR52/55/56/35, HBB/HBA1/HBA2, APOE/APOC1, SLC6A4/SLC6A3) so that a
pipeline still generating symbols will visibly fail here.

```bash
python ../scripts/check_tokenizer.py --model <model-id> \
    --genes GPR52 GPR56 HBB HBA1

python ../scripts/score_pmi.py --model <model-id> \
    --genes candidates.txt --diseases diseases.txt \
    --out stage1.jsonl --neutral-cache neutral.npz

python ../scripts/evaluate.py --pred stage1.jsonl --gold gold.jsonl \
    --k 1 3 5 --controls
```

Four diseases is far too few to conclude anything. Expand to 100+ known pairs
before reading the numbers as real.
