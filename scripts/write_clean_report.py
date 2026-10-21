"""导出完整 109 类同集对比，以及可公开提交的数值报告。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from local_experiment import ROOT

def read(path):
    return json.loads(path.read_text())

def predictions(run, name):
    return {r['hash']:r for r in (json.loads(s) for s in (run/f'{name}_predictions.jsonl').read_text().splitlines())}

def main():
    from sklearn.metrics import classification_report,accuracy_score,f1_score
    folder = ROOT/'reports'
    folder.mkdir(exist_ok=True)
    old,new = ROOT/'runs/clean_baseline',ROOT/'runs/clean_lora_109'
    before,after = read(old/'result.json'),read(new/'result.json')
    data = read(ROOT/'data/clean_round/summary.json')
    audit = read(ROOT/'data/clean_round/split_audit.json')
    cfg = read(new/'config.json')
    names = {r['code']:r['语言名称'] for r in csv.DictReader((ROOT/'languages.csv').open())}
    labels = sorted(names)
    base,pred = predictions(old,'test'),predictions(new,'test')
    assert {h:r['true'] for h,r in base.items()}=={h:r['true'] for h,r in pred.items()}
    assert len(base)==10900
    excluded = set(audit['exclude_short_hashes'])
    def short_clean(run):
        rows = [r for h,r in predictions(run,'test_first_40_characters').items() if h not in excluded]
        y,p = [r['true'] for r in rows],[r['predicted'] for r in rows]
        report = classification_report(y,p,labels=labels,output_dict=True,zero_division=0)
        return {'samples':len(rows),'accuracy':accuracy_score(y,p),
                'macro_f1':f1_score(y,p,labels=labels,average='macro',zero_division=0),
                'icelandic':report['isl']},report
    old_short,old_short_report = short_clean(old)
    new_short,new_short_report = short_clean(new)
    changes = Counter()
    for h in base:
        a = base[h]['predicted']==base[h]['true']
        b = pred[h]['predicted']==pred[h]['true']
        changes['fixed' if b and not a else 'regressed' if a and not b else 'unchanged'] += 1
    results = {'before':before,'after':after,'before_short_clean':old_short,'after_short_clean':new_short,
      'paired_changes':dict(changes),'test_languages':109,'test_samples_per_language':100,
      'short_exclusions':len(excluded),'training_samples':data['samples']['train'],
      'training_synthetic':data['training_synthetic'],'removed':data['removed']}
    (folder/'latest_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    details_before,details_after = read(old/'test_per_language.json'),read(new/'test_per_language.json')
    with (folder/'per_language.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['code','language','support','before_f1','after_f1','delta_pp','short_clean_support','before_short_f1','after_short_f1'])
        for label in labels:
            a,b=details_before[label],details_after[label]
            assert a['support']==b['support']==100
            w.writerow([label,names[label],100,a['f1-score'],b['f1-score'],
                        100*(b['f1-score']-a['f1-score']),int(new_short_report[label]['support']),
                        old_short_report[label]['f1-score'],new_short_report[label]['f1-score']])
    with (folder/'test_sources.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['label','dataset','variant','revision','url','license'],lineterminator='\n')
        w.writeheader();w.writerows(data['test_sources'])
    (folder/'data_audit.json').write_text(json.dumps({k:v for k,v in audit.items() if k!='exclude_short_hashes'},indent=2))
    manifest = {'status':'complete','base_revision':cfg['base_revision'],
      'parent_adapter_sha256':cfg['initial_adapter_sha256'],
      'adapter_sha256':hashlib.file_digest((new/'best_adapter/adapter_model.safetensors').open('rb'),'sha256').hexdigest(),
      'input_files_sha256':read(new/'prepared_fingerprints.json'),
      'test_content_labels_identical_before_after':True,'trained_epochs':cfg['epochs'],
      'test_sources':data['test_sources'],'manual_validation':'see local run artifacts, excluded from git'}
    (folder/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    pct=lambda v:f'{100*v:.2f}%'
    metrics=[('原文 Macro-F1',before['test']['macro_f1'],after['test']['macro_f1']),
      ('前40字符 Macro-F1（原始）',before['short_text']['macro_f1'],after['short_text']['macro_f1']),
      ('前40字符 Macro-F1（排除输入重合）',old_short['macro_f1'],new_short['macro_f1']),
      ('冰岛语原文 F1',before['test']['icelandic']['f1-score'],after['test']['icelandic']['f1-score']),
      ('冰岛语前40字符 F1（排除重合）',old_short['icelandic']['f1-score'],new_short['icelandic']['f1-score'])]
    lines=['# 清洗后追加一轮：109 类完整评估','',
      '使用同一套全新测试文本对比上一版与清洗后追加一轮的模型。所有数值来自本地 MPS 实测。','',
      '## 同集结果','',
      '| 指标 | 上一版 | 清洗并追加一轮 | 变化（百分点） |','|---|---:|---:|---:|']
    for title,a,b in metrics:
        lines.append(f'| {title} | {pct(a)} | {pct(b)} | {(b-a)*100:+.2f} |')
    lines += ['',f'原文测试共 10,900 条，109 类各 100 条。剔除输入重合后的短文本测试为 {new_short["samples"]:,} 条，'
      f'排除 {len(excluded)} 条；前后两模型使用完全相同的样本和标签。', '',
      '变化按未四舍五入的原始数值计算。该轮不是全面提升：短文本总体小幅上升，原文总体基本持平，'
      '冰岛语短文本回退。默认推理仍保留上一版 expanded_lora_109；clean_lora_109 作为候选对照，不宣称取代默认。','',
      f'按原文预测逐条配对，本轮纠正 {changes["fixed"]} 条，同时有 {changes["regressed"]} 条从正确变错误。', '',
      '之前 88 类 OPUS 测试上的 88.17% 不能与这里的新分数直接相减。原先来源标签存在噪声、'
      '语言覆盖与体裁也不同。只能以上表的同集差值作为这轮变化。','',
      '## 补齐的测试与数据映射','',
      '106 类采用 FLORES devtest，布列塔尼语、西弗里斯兰语、拉丁语采用独立 UD 树库测试文本。'
      '剩余 21 类已全部有测试样本；没有把合成文本当成测试答案。新测试未用于训练、学习率选择或教师筛选。','',
      'FLORES 采用部分标准变体映射到模型的宏语言标签，例如阿拉伯语为标准阿拉伯语、蒙古语为喀尔喀蒙古语；'
      '中文样本包含简繁，输出仍为一个 zho 类。覆盖一个变体不等于所有方言、转写和书写系统都通过。'
      '逐类来源、变体、固定提交和许可证见 [test_sources.csv](test_sources.csv)。','',
      '来源：[FLORES](https://huggingface.co/datasets/mteb/flores)、'
      '[Latin LLCT](https://github.com/UniversalDependencies/UD_Latin-LLCT)、'
      '[Breton KEB](https://github.com/UniversalDependencies/UD_Breton-KEB)、'
      '[Frisian Frysk](https://github.com/UniversalDependencies/UD_Frisian-Frysk/tree/dev)。'
      '这次采用的测试资源均为 CC BY-SA 4.0，仓库只提交数值与来源，不分发语料。','',
      '## 清洗与本地成本','',
      f'- 训练从 59,675 条减为 {data["samples"]["train"]:,} 条，验证从 5,000 条减为 {data["samples"]["validation"]:,} 条。',
      f'- 训练包含 {data["training_synthetic"]:,} 条既有增强样本；只保留仍有对应真实训练父样本的衍生项。',
      '- 清理复制英文的平行语料、文字脚本明显不符和论坛时间模板；不强行重标近缘语言。',
      '- 历史实验中的文本与完整模型输入被排除出新测试；当前训练/验证/测试完整输入无重合。'
      '派生短文本的输入重合另行剔除，不掩盖这类差异。',
      '- 初始化自上一版适配器，固定种子、按语言均衡采样，LoRA r=8，batch 32、96 token，追加 1 epoch。',
      f'- M2 Max / 32GB，训练及验证 {after["training_and_validation_seconds"]/60:.2f} 分钟；'
      f'包含两组测试共 {after["total_seconds"]/60:.2f} 分钟，不含下载与备数。','',
      '删除计数：`'+json.dumps(data['removed'],ensure_ascii=False)+'`。','',
      '## 较低语言与限制','',
      '| 语言 | 上一版 F1 | 本轮 F1 | 本轮去重短文本 F1 |','|---|---:|---:|---:|']
    for label in sorted(labels,key=lambda s:details_after[s]['f1-score'])[:12]:
        lines.append(f'| {names[label]}（{label}） | {pct(details_before[label]["f1-score"])} | '
          f'{pct(details_after[label]["f1-score"])} | {pct(new_short_report[label]["f1-score"])} |')
    lines += ['', '本轮是固定种子的一次试验；每类只有 100 条原文测试，不能据微小分差断言稳定业务提升。'
      '测试语言标签来自翻译/树库来源，仍可能包含借词、近缘语言歧义或标注问题；'
      '没有用本模型预测修正测试标签。未验证真实业务分布、未知语言、混合语言拒识或置信度校准。'
      '已检查的文本隔离不等于完成所有近重复聚类，也无法排除基座预训练污染。','',
      '本机已验证能完成当前规模 LoRA 训练。总体均分不能替代逐语言验收；'
      '完整结果见 [per_language.csv](per_language.csv)，指标与溯源见 [latest_metrics.json](latest_metrics.json)'
      ' 和 [experiment_manifest.json](experiment_manifest.json)。','']
    report='\n'.join(lines)
    (folder/'clean_round_report.md').write_text(report,encoding='utf-8')
    (ROOT/'local_evaluation_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'metrics':metrics,'paired_changes':dict(changes),'short_samples':new_short['samples']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
