"""清理平行语料错标，构建不经模型筛选的 109 类完整新测试集。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import unicodedata
from local_experiment import ROOT, LANGUAGES
from prepare_expanded import norm, fingerprint

FLORES_REV = '69edd3336d102dd184dd531b57defac6c9c60c8f'
TREEBANKS = {
    'lat': ('UD_Latin-LLCT', '173cd1133e70f1a95a8608091792f91ebe001157', 'la_llct-ud-test.conllu'),
    'bre': ('UD_Breton-KEB', '639600f9b15a5e2e15f310b12e4dbe81c40fa74b', 'br_keb-ud-test.conllu'),
    'fry': ('UD_Frisian-Frysk', 'ed48f6668df7a13568524f809cbc1cda8d613b4a', 'fy_frysk-ud-test.conllu'),
}
ALIASES = {'ara':'arb','aze':'azj','fas':'pes','kur':'kmr','lav':'lvs','mlg':'plt',
           'mon':'khk','msa':'zsm','nep':'npi','ori':'ory','orm':'gaz','pus':'pbt',
           'sqi':'als','swa':'swh','uzb':'uzn','yid':'ydd'}

def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + '.partial')
        subprocess.run(['curl','-fsSL','--retry','2','--connect-timeout','15','--max-time','120',
                        '-o',str(part),url],check=True)
        part.rename(path)

def tokenize_hashes(tokenizer, rows):
    keys = set()
    for offset in range(0,len(rows),512):
        ids = tokenizer(rows[offset:offset+512],truncation=True,max_length=96)['input_ids']
        keys.update(hashlib.sha256(json.dumps(v).encode()).hexdigest() for v in ids)
    return keys

def conllu_texts(path):
    for block in path.read_text().split('\n\n'):
        text = next((line[9:] for line in block.splitlines() if line.startswith('# text = ')),None)
        if text is None:
            tokens = [line.split('\t') for line in block.splitlines() if not line.startswith('#') and '\t' in line]
            text = ' '.join(t[1] for t in tokens if t[0].isdigit())
        if sum(c.isalpha() for c in text) >= 10:
            yield norm(text)

def scripts(label):
    special = {'amh':['ETHIOPIC'],'ara':['ARABIC'],'fas':['ARABIC'],'pus':['ARABIC'],
      'snd':['ARABIC'],'uig':['ARABIC'],'urd':['ARABIC'],'ckb':['ARABIC'],
      'asm':['BENGALI'],'ben':['BENGALI'],'bod':['TIBETAN'],'guj':['GUJARATI'],
      'hin':['DEVANAGARI'],'mar':['DEVANAGARI'],'nep':['DEVANAGARI'],'san':['DEVANAGARI'],
      'pan':['GURMUKHI','ARABIC'],'tam':['TAMIL'],'tel':['TELUGU'],'kan':['KANNADA'],
      'mal':['MALAYALAM'],'ori':['ORIYA'],'sin':['SINHALA'],'tha':['THAI'],'lao':['LAO'],
      'khm':['KHMER'],'mya':['MYANMAR'],'kor':['HANGUL','CJK'],'jpn':['HIRAGANA','KATAKANA','CJK'],
      'zho':['CJK','IDEOGRAPH'],'kat':['GEORGIAN'],'ell':['GREEK'],'heb':['HEBREW'],
      'yid':['HEBREW'],'hye':['ARMENIAN'],'srp':['CYRILLIC','LATIN'],
      'aze':['LATIN','CYRILLIC','ARABIC'],'kaz':['CYRILLIC','LATIN','ARABIC'],
      'kur':['LATIN','ARABIC'],'mon':['CYRILLIC','MONGOLIAN']}
    if label in special:
        return special[label]
    if label in {'bel','bul','rus','ukr','kir','mkd','tat','tgk'}:
        return ['CYRILLIC']
    return ['LATIN']

def main():
    import fasttext
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    folder = ROOT / 'data/clean_round'
    folder.mkdir(parents=True,exist_ok=True)
    urls = {}
    flores = ROOT / 'data/flores_devtest.parquet'
    urls['flores'] = f'https://huggingface.co/datasets/mteb/flores/resolve/{FLORES_REV}/devtest.parquet'
    fetch(urls['flores'],flores)
    for label,(repo,revision,file) in TREEBANKS.items():
        base = f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/{revision}'
        urls[label] = base + '/' + file
        fetch(urls[label],ROOT/'data/ud_test'/label/file)
        fetch(base+'/LICENSE.txt',ROOT/'data/ud_test'/label/'license.txt')
    print('测试资源已就绪，排除历史实验已见文本及输入',flush=True)
    old = {split:[json.loads(s) for s in (ROOT/'data/expanded'/f'{split}.jsonl').read_text().splitlines()]
           for split in ['train','validation','test']}
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/xlm_roberta_base',local_files_only=True)
    previous = {fingerprint(r['text']):r['text'] for rows in old.values() for r in rows}
    wanted = set()
    for split in ['train','validation','test']:
        wanted.update(json.loads(s)['hash'] for s in (ROOT/'runs/local_lora_109'/f'{split}_manifest.jsonl').read_text().splitlines())
    for split in ['train','test']:
        for row in pq.read_table(ROOT/'data'/f'{split}.parquet',columns=['sentence']).to_pylist():
            h = fingerprint(row['sentence'])
            if h in wanted:
                previous[h] = norm(row['sentence'])
    forbidden_hashes = set(previous)
    forbidden_tokens = tokenize_hashes(tokenizer,list(previous.values()))
    print(json.dumps({'historical_unique_texts':len(previous),'historical_inputs':len(forbidden_tokens)}),flush=True)
    candidates = {}
    table = pq.read_table(flores)
    mapping = []
    rng = random.Random(20261022)
    for label in sorted(LANGUAGES):
        if label in TREEBANKS:
            repo,revision,file = TREEBANKS[label]
            candidates[label] = [(text,f'{repo}:{i}') for i,text in enumerate(conllu_texts(ROOT/'data/ud_test'/label/file))]
            mapping.append({'label':label,'dataset':repo,'variant':'Latin script',
                            'revision':revision,'url':urls[label],'license':'CC BY-SA 4.0'})
        else:
            prefix = ALIASES.get(label,label)
            columns = sorted(c for c in table.column_names if c.split('_')[0]==prefix)
            assert columns,(label,prefix)
            if label=='zho':
                columns = [c for c in columns if c in {'zho_Hans','zho_Hant'}]
            else:
                columns = columns[:1]
            rows = table.select(columns).to_pylist()
            candidates[label] = [(norm(row[c]),f'flores:{i}') for i,row in enumerate(rows) for c in columns]
            mapping.append({'label':label,'dataset':'mteb/flores','variant':','.join(columns),
                            'revision':FLORES_REV,'url':urls['flores'],'license':'CC BY-SA 4.0'})
        rng.shuffle(candidates[label])
    used_hash,used_tokens = set(),set()
    test = []
    for label in sorted(LANGUAGES):
        selected = []
        parents = set()
        for text,parent in candidates[label]:
            h = fingerprint(text)
            if h in forbidden_hashes or h in used_hash or parent in parents:
                continue
            key = next(iter(tokenize_hashes(tokenizer,[text])))
            if key in forbidden_tokens or key in used_tokens:
                continue
            selected.append({'text':text,'label':label,'hash':h,'parent':parent,
                             'source':'fresh_flores_ud','synthetic':False})
            used_hash.add(h);used_tokens.add(key);parents.add(parent)
            if len(selected)==100:
                break
        assert len(selected)==100,(label,len(selected))
        test.extend(selected)
    print(json.dumps({'fresh_test_samples':len(test),'fresh_test_languages':len(set(r['label'] for r in test))}),flush=True)

    pairs = {}
    for path in sorted((ROOT/'data/opus_file_samples').glob('*/*.json')):
        payload = json.loads(path.read_text())
        config,split = path.parent.name,path.stem
        for item in payload['rows']:
            pairs[f'opus:{config}:{split}:{item["row_idx"]}'] = item['row']['translation']
    teacher = fasttext.load_model(str(ROOT/'models/lid.176.ftz'))
    english_words = set('the a an you and of to in for with this that it your not on are be from have is'.split())
    removed = Counter()
    def reason(record):
        text,label = record['text'],record['label']
        pair = pairs.get(record['parent'])
        if label!='eng' and pair:
            en = ''.join(c for c in norm(pair['en']).casefold() if c.isalnum())
            foreign = next(v for k,v in pair.items() if k!='en')
            other = ''.join(c for c in norm(foreign).casefold() if c.isalnum())
            if en==other:
                return 'untranslated_parallel_copy'
        alpha = [c for c in text if c.isalpha()]
        expected = scripts(label)
        if alpha and sum(any(s in unicodedata.name(c,'') for s in expected) for c in alpha)/len(alpha)<.5:
            return 'script_mismatch'
        if re.search(r'\bon:\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',text,re.I):
            return 'forum_time_template'
        if re.search(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b',text) and sum(c.isdigit() for c in text)>8:
            return 'forum_time_template'
        if label!='eng':
            words = re.findall(r'[a-z]+',text.casefold())
            hits = set(words)&english_words
            if len(hits)>=4 and sum(w in english_words for w in words)/max(1,len(words))>=.35:
                pred = teacher.f.predict(norm(text)+'\n',1,0.,'strict')
                if pred and pred[0][1]=='__label__en' and pred[0][0]>=.95:
                    return 'clear_english_under_other_label'
        return None
    result = {'train':[],'validation':[],'test':test}
    kept_parents = set()
    for record in old['train']:
        if record['synthetic']:
            continue
        why = reason(record)
        if why:
            removed[why] += 1
        else:
            result['train'].append(record)
            kept_parents.add((record['parent'],record['label']))
    for record in old['train']:
        if record['synthetic']:
            if (record['parent'],record['label']) not in kept_parents:
                removed['derivative_of_removed_parent'] += 1
            else:
                result['train'].append(record)
    for record in old['validation']:
        why = reason(record)
        if why:
            removed['validation_'+why] += 1
        else:
            result['validation'].append(record)
    assert set(r['label'] for r in result['train'])==set(LANGUAGES)
    for split,rows in result.items():
        rng.shuffle(rows)
        with (folder/f'{split}.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts = {split:dict(Counter(r['label'] for r in rows)) for split,rows in result.items()}
    summary = {'samples':{s:len(v) for s,v in result.items()},'counts':counts,
      'training_languages':109,'test_languages':109,'removed':dict(removed),
      'training_synthetic':sum(r['synthetic'] for r in result['train']),
      'test_selection':'100 per language; no teacher predictions used for test filtering or relabeling',
      'heldout_history_excluded':len(forbidden_tokens),
      'test_sources':mapping,'source_sha256':{str(flores.relative_to(ROOT)):hashlib.file_digest(flores.open('rb'),'sha256').hexdigest()}}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:summary[k] for k in ['samples','training_languages','test_languages','removed','training_synthetic']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
