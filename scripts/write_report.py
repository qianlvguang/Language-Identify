"""从实际训练、评测和去重审计产物生成中文实验报告。"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/local_lora_109')
    args = parser.parse_args()
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    run = args.run
    raw = read(run / 'result.json')
    labels = read(run / 'languages.json')
    config = read(run / 'config.json')
    audit = read(run / 'split_audit.json')
    counts = read(run / 'data_counts.json')
    def clean(name, split):
        excluded = set(audit['exclude_for_clean_evaluation'][split])
        rows = [json.loads(s) for s in (run / f'{name}_predictions.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['hash'] not in excluded]
        y = [r['true'] for r in rows]
        p = [r['predicted'] for r in rows]
        report = classification_report(y, p, labels=labels, target_names=labels,
                                       output_dict=True, zero_division=0)
        metrics = {'samples': len(rows), 'accuracy': accuracy_score(y, p),
                   'macro_f1': f1_score(y, p, labels=labels, average='macro', zero_division=0),
                   'icelandic': report['isl']}
        (run / f'{name}_clean_summary.json').write_text(json.dumps(metrics, indent=2))
        (run / f'{name}_clean_per_language.json').write_text(json.dumps(report, indent=2))
        return metrics, report, rows
    validation = [clean(f'validation_epoch_{i+1}', 'validation')[0] for i in range(config['epochs'])]
    original_best = max(range(len(raw['validation_history'])),
                        key=lambda i: raw['validation_history'][i]['macro_f1']) + 1
    clean_best = max(range(len(validation)), key=lambda i: validation[i]['macro_f1']) + 1
    test, per_language, predictions = clean('test', 'test')
    short, _, _ = clean('test_first_40_characters', 'test')
    clean_result = {'validation': validation, 'original_selected_epoch': original_best,
                    'clean_selected_epoch': clean_best, 'selection_unchanged': original_best == clean_best,
                    'test': test, 'short_text': short}
    (run / 'clean_result.json').write_text(json.dumps(clean_result, indent=2))
    with (run / 'per_language.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['language', 'precision', 'recall', 'f1', 'test_samples'])
        for label in labels:
            r = per_language[label]
            writer.writerow([label, r['precision'], r['recall'], r['f1-score'], int(r['support'])])
    logs = [json.loads(s) for s in (run / 'training.jsonl').read_text().splitlines()]
    step_times = [(b['elapsed_seconds'] - a['elapsed_seconds']) / (b['step'] - a['step'])
                  for a, b in zip(logs, logs[1:]) if a['epoch'] == b['epoch'] and a['step'] >= 100]
    median = statistics.median(step_times)
    lowest = sorted(labels, key=lambda s: per_language[s]['f1-score'])[:12]
    pct = lambda n: f'{100*n:.2f}%'
    sizes = {split: sum(row[split] for row in counts.values()) for split in ['train', 'validation', 'test']}
    errors = Counter(r['predicted'] for r in predictions if r['true'] == 'isl' and r['predicted'] != 'isl')
    lines = ['# 本地语言识别训练评估报告', '',
             '本报告由实际运行产物生成；训练使用 M2 Max 的 MPS，未使用远端 GPU。', '',
             '## 结果', '',
             '| 测试方式 | 条数 | 准确率 | Macro-F1 | 冰岛语 F1 |',
             '|---|---:|---:|---:|---:|']
    for name, metric in [('原文，最多 96 token，去除输入重合', test),
                          ('前 40 字符派生短文本，去除同一重合项', short)]:
        lines.append(f'| {name} | {metric["samples"]} | {pct(metric["accuracy"])} | {pct(metric["macro_f1"])} | {pct(metric["icelandic"]["f1-score"])} |')
    lines += ['', f'覆盖 {len(labels)} 类；实际训练 / 验证 / 测试样本为 '
              f'{sizes["train"]:,} / {sizes["validation"]:,} / {sizes["test"]:,}。'
              '除海地克里奥尔语去重后样本较少外，各类训练 200、验证 50、测试 100。', '',
              f'冰岛语原文测试：precision {pct(test["icelandic"]["precision"])}、'
              f'recall {pct(test["icelandic"]["recall"])}，样本 {int(test["icelandic"]["support"])} 条；'
              f'错误去向计数：{dict(errors)}。100 条测试的统计把握有限，不能据此宣称接近零错误。', '',
              '## 硬件与训练成本', '',
              f'- M2 Max，30 核 GPU，32GB 统一内存；PyTorch {config["torch"]}，MPS FP32。',
              f'- 总参数 {config["total"]:,}，可训练参数 {config["trainable"]:,}；LoRA r=8，加可训练分类头。',
              f'- batch {config["batch_size"]}，最大长度 {config["max_length"]} token，{config["epochs"]} epoch。',
              f'- 稳态记录区间中位数 {median:.3f} 秒/步，约 {config["batch_size"]/median:.1f} 条/秒。',
              f'- 训练及每轮验证合计 {raw["training_and_validation_seconds"]/60:.2f} 分钟；'
              f'含最终两组测试共 {raw["total_seconds"]/60:.2f} 分钟，不含下载和导入准备。',
              f'- 采样记录中 MPS 驱动分配最大 {max(r["mps_driver_gb"] for r in logs):.2f} GB，'
              f'进程 RSS 最大 {max(r["rss_gb"] for r in logs):.2f} GB；二者不可相加作为实际物理占用。',
              f'- 系统可用内存最低 {min(r["system_available_gb"] for r in logs):.2f} GB；'
              f'交换空间使用最大 {max(r["swap_used_gb"] for r in logs):.2f} GB。以上是间隔采样值，不是精确瞬时峰值。', '',
              '## 验证集与隔离检查', '',
              '| 轮次 | 去除重合后的验证 Macro-F1 |', '|---|---:|']
    for i, r in enumerate(validation):
        lines.append(f'| {i+1} | {pct(r["macro_f1"])} |')
    lines += ['', f'原始验证选中第 {original_best} 轮；剔除重合项后选中第 {clean_best} 轮。'
              + ('选择一致。' if original_best == clean_best else '选择不一致，当前适配器不应视为清洁验证集最优，需要重新选取 checkpoint。'), '',
              f'完整文本训练/验证/测试重合计数为 {audit["full_text_overlap"]}。96 token 截断后'
              f'训练与验证重合 {audit["truncated_input_overlap"]["train/validation"]} 项、'
              f'训练与测试重合 {audit["truncated_input_overlap"]["train/test"]} 项；'
              f'验证与测试重合 {audit["truncated_input_overlap"]["validation/test"]} 项。'
              f'干净测试剔除 {len(audit["exclude_for_clean_evaluation"]["test"])} 条与训练或验证输入重合的样本。'
              '训练模型本身未据本轮测试结果调参。'
              '仍未进行近重复文档聚类，也无法排除基座预训练数据污染。', '',
              '## 原文测试较低的语言', '',
              '| 语言代码 | Precision | Recall | F1 | 样本 |', '|---|---:|---:|---:|---:|']
    for label in lowest:
        r = per_language[label]
        lines.append(f'| {label} | {pct(r["precision"])} | {pct(r["recall"])} | {pct(r["f1-score"])} | {int(r["support"])} |')
    long_pass = test['macro_f1'] >= .95 and test['icelandic']['f1-score'] >= .95
    lines += ['', '## 是否足够', '',
              f'硬件：这台 Mac 能完成本轮 {len(labels)} 类 LoRA 训练，内存和速度均有实际记录支持。'
              '这不能直接外推为全参数微调或长序列训练的结论。', '',
              ('同域效果达到本轮暂定 Macro-F1 和冰岛语 F1 均不低于 95% 的参考线。' if long_pass else
               '同域效果尚未达到本轮暂定 Macro-F1 和冰岛语 F1 均不低于 95% 的参考线。'),
              '是否满足业务需求仍未确定：目前只有百科同域测试，40 字符版本是派生测试；'
              '尚缺真实业务短文本、独立来源、未知语言和混合语言拒识评测。'
              '置信度未校准，预测脚本输出的 score 不应解释为真实正确率。', '',
              '扩大训练前应优先补真实短文本、人工检查的近缘语言和独立测试集。'
              f'若扩到每语言 10,000 条，{len(labels)} 类共 {len(labels)} 万条，3 轮训练量约为当前的 50 倍；'
              '仅能按本次吞吐估算同配置量级，实际仍受序列长度、数据管道、散热和采样策略影响。', '',
              '## 产物', '',
              '- `best_adapter/`：选定 LoRA 与分类头；需要项目内本地 XLM-R 基座配合推理。',
              '- `clean_result.json`：剔除输入重合后的汇总。',
              f'- `per_language.csv`：{len(labels)} 类完整分语言结果。',
              '- `test_confusion.csv`：原始完整测试混淆矩阵；干净测试预测由对应清单排除重合项后得到。',
              '- `split_audit.json`、三个 `*_manifest.jsonl`：隔离审计和数据指纹。',
              '- `training.jsonl`、`config.json`、`requirements.lock.txt`（项目根目录）：运行与环境证据。',
              '', '数据来源：[WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018)；'
              '基座：[XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)。', '']
    (ROOT / 'local_evaluation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(clean_result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
