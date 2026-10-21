"""不用下载模型，检查公开仓库的覆盖声明与发布边界。"""
import ast
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    tree = ast.parse((ROOT/'scripts/local_experiment.py').read_text())
    labels = next(ast.literal_eval(n.value.func.value).split() for n in tree.body
                  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LANGUAGES' for t in n.targets))
    names = list(csv.DictReader((ROOT/'languages.csv').open()))
    assert len(labels)==len(set(labels))==len(names)==109
    assert set(labels)=={r['code'] for r in names}
    assert {'eng','zho','spa','isl'}<=set(labels)
    release = json.loads((ROOT/'reports/latest_metrics.json').read_text())
    assert release['after']['test']['evaluated_languages']==109
    assert release['after']['test']['missing_languages']==[]
    assert release['after']['test']['samples']==10900
    per_language = list(csv.DictReader((ROOT/'reports/per_language.csv').open()))
    assert {r['code'] for r in per_language}==set(labels)
    assert all(int(r['support'])==100 for r in per_language)
    tracked = subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    forbidden = ['data/','models/','runs/','.venv/','.env']
    assert not any(any(p.startswith(prefix) for prefix in forbidden) for p in tracked)
    assert not any(p.endswith(('.safetensors','.parquet','.ftz','.conllu')) for p in tracked)
    print('109 类、10,900 条测试、逐语言指标与发布文件边界检查通过；不代表 GPU 训练验证。')

if __name__=='__main__':
    main()
