"""独立复核增强数据的输入、父样本隔离及派生短文本重合。"""
import hashlib
import argparse
import json
from pathlib import Path
from local_experiment import ROOT

def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/expanded')
    args = parser.parse_args()
    folder = args.data_dir
    data = {split: [json.loads(s) for s in (folder / f'{split}.jsonl').read_text().splitlines()]
            for split in ['train', 'validation', 'test']}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/xlm_roberta_base', local_files_only=True)
    keys = {}
    for split, rows in data.items():
        keys[split] = set()
        for start in range(0, len(rows), 512):
            batch = rows[start:start+512]
            ids = tokenizer([r['text'] for r in batch], truncation=True, max_length=96)['input_ids']
            keys[split].update(hashlib.sha256(json.dumps(v).encode()).hexdigest() for v in ids)
    overlaps = {}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlaps[a + '/' + b] = len(keys[a] & keys[b])
        assert overlaps[a + '/' + b] == 0
        assert not ({r['parent'] for r in data[a]} & {r['parent'] for r in data[b]})
    real_parents = {r['parent'] for r in data['train'] if not r['synthetic']}
    assert all(r['parent'] in real_parents for r in data['train'] if r['synthetic'])
    assert all(not r['synthetic'] for split in ['validation', 'test'] for r in data[split])
    short_overlap = []
    forbidden = keys['train'] | keys['validation']
    for start in range(0, len(data['test']), 512):
        batch = data['test'][start:start+512]
        encoded = tokenizer([r['text'][:40] for r in batch], truncation=True, max_length=96)['input_ids']
        for row, ids in zip(batch, encoded):
            if hashlib.sha256(json.dumps(ids).encode()).hexdigest() in forbidden:
                short_overlap.append(row['hash'])
    result = {'full_input_overlap': overlaps,
              'all_synthetic_parents_in_real_training': True,
              'no_synthetic_evaluation': True,
              'short_derived_input_overlap_count': len(short_overlap),
              'exclude_short_hashes': short_overlap}
    (folder / 'split_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
