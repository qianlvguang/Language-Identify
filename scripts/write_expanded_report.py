"""在同一新增测试集上比较基线与扩充训练，生成可核查的中文报告。"""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    run = ROOT / 'runs/expanded_lora_109'
    baseline = ROOT / 'runs/expanded_baseline'
    before, after = read(baseline / 'result.json'), read(run / 'result.json')
    def truth_map(folder):
        return {r['hash']: r['true'] for r in
                (json.loads(s) for s in (folder / 'test_predictions.jsonl').read_text().splitlines())}
    assert truth_map(baseline) == truth_map(run), '前后比较必须使用完全相同的测试样本与标签'
    data = read(ROOT / 'data/expanded/summary.json')
    audit = read(ROOT / 'data/expanded/split_audit.json')
    cfg = read(run / 'config.json')
    logs = [json.loads(s) for s in (run / 'training.jsonl').read_text().splitlines()]
    names = {r['code']: r['语言名称'] for r in csv.DictReader((ROOT / 'languages.csv').open())}
    pct = lambda v: f'{100*v:.2f}%'
    lines = ['# XLM-R 本地语言识别：扩充数据后的评估', '',
             '结论依据真实 M2 Max / MPS 训练和同一组新增测试句；合成样本不参与验证或测试。', '',
             '## 同一测试集上的前后对比', '',
             '| 指标 | 原 109 类基线 | 扩充训练后 | 变化（百分点） |', '|---|---:|---:|---:|']
    metrics = [('原始句子 Macro-F1', 'test', None), ('前 40 字符 Macro-F1', 'short_text', None),
               ('冰岛语原始句子 F1', 'test', 'icelandic'), ('冰岛语前 40 字符 F1', 'short_text', 'icelandic')]
    for title, split, sub in metrics:
        a = before[split][sub]['f1-score'] if sub else before[split]['macro_f1']
        b = after[split][sub]['f1-score'] if sub else after[split]['macro_f1']
        lines.append(f'| {title} | {pct(a)} | {pct(b)} | {(b-a)*100:+.2f} |')
    lines += ['', f'测试包含 {after["test"]["samples"]:,} 条真实句子，覆盖 {after["test"]["evaluated_languages"]} 种语言；'
              f'分类器仍输出全部 {cfg["languages"]} 类。Macro-F1 仅对本次有真实测试样本的语言求平均，'
              '不能解释为全部 109 类均已达到这个水平。', '',
              '尚无本次外部测试的语言：' + '、'.join(f'{names[s]}（{s}）' for s in data['missing_external_test']) + '。', '',
              '## 实际训练数据与合成方式', '',
              f'- 训练 {data["samples"]["train"]:,} 条、验证 {data["samples"]["validation"]:,} 条、'
              f'测试 {data["samples"]["test"]:,} 条。训练覆盖 {data["training_languages"]} 类。',
              f'- 其中训练增强/合成 {data["synthetic_training"]:,} 条，其余来自真实语料。',
              '- 新增真实句子来自 OPUS-100 翻译句对，按原始语言侧标注，按固定提交读取分散行组。',
              '- 用独立 FastText 删除训练中高置信度（≥0.75）、非近缘的标签冲突；保留不确定项和近缘类别的来源标签。教师不重标、不筛测试，也不等同人工金标。',
              '- 合成方式为训练父文本的随机短片段、标点/数字轻扰动，以及已知旧字幕编码变体。每条均保留 parent、source 和 synthetic 字段。',
              '- 冰岛语训练端修复旧字幕的 đ/ū/ķ/ũ 等错误字形，同时将相应旧字形保留为训练增强；测试保持原始数据，不按正确答案修复。',
              '- 全文规范化哈希、96-token 输入哈希和父样本均检查隔离；新增保留集排除基线实验已经使用的文本和相同输入。', '',
              f'独立审计：模型完整输入重合 {audit["full_input_overlap"]}；'
              f'派生短文本与训练/验证输入重合 {audit["short_derived_input_overlap_count"]} 条；'
              '所有增强样本均有真实训练父样本，验证和测试均无增强样本。', '',
              '## 本机训练成本', '',
              f'- M2 Max、30 核 GPU、32GB 统一内存；MPS FP32，LoRA r=8。',
              f'- 从原 109 类最佳适配器继续训练 {cfg["epochs"]} 轮，分类标签保持一致；按新增验证集选 checkpoint。',
              f'- 新验证覆盖 {after["validation_history"][0]["evaluated_languages"]} 类；'
              f'缺少新验证的类别为 {after["validation_history"][0]["missing_languages"]}，不据此宣称这些类别达标。',
              f'- 可训练参数 {cfg["trainable"]:,}，batch {cfg["batch_size"]}，最大 {cfg["max_length"]} token。',
              '- 按语言逆频率有放回采样，减轻不同语言真实数据量不均衡的影响。',
              f'- 本次训练及验证 {after["training_and_validation_seconds"]/60:.2f} 分钟，'
              f'含最终两组测试共 {after["total_seconds"]/60:.2f} 分钟；不含基线训练、数据准备和下载。',
              f'- 采样 MPS 驱动分配最大 {max(r["mps_driver_gb"] for r in logs):.2f} GB；'
              f'系统可用内存最低 {min(r["system_available_gb"] for r in logs):.2f} GB；'
              f'交换空间使用最大 {max(r["swap_used_gb"] for r in logs):.2f} GB。不是精确瞬时峰值。', '',
              '## 分语言结果', '']
    details = read(run / 'test_per_language.json')
    short = read(run / 'test_first_40_characters_per_language.json')
    supported = [s for s in names if details[s]['support'] > 0]
    lower = sorted(supported, key=lambda s: details[s]['f1-score'])[:12]
    with (run / 'per_language.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['code', '语言', '样本', '原文F1', '40字符F1'])
        for s in names:
            writer.writerow([s, names[s], int(details[s]['support']), details[s]['f1-score'] if details[s]['support'] else '',
                             short[s]['f1-score'] if short[s]['support'] else ''])
    lines += ['原始句子较低的类别：', '', '| 语言 | 样本 | 原文 F1 | 40 字符 F1 |', '|---|---:|---:|---:|']
    for s in lower:
        lines.append(f'| {names[s]}（{s}） | {int(details[s]["support"])} | {pct(details[s]["f1-score"])} | {pct(short[s]["f1-score"])} |')
    below = [s for s in supported if details[s]['f1-score'] < .9]
    lines += ['', '## 是否足够与下一步', '',
              '硬件方面：本机可以实际完成 109 类 LoRA 训练和评测；当前无需因为显存容量而换卡。'
              '百万条规模或全参数训练的速度仍需另测。', '',
              f'效果方面：本次有外部测试的语言中，仍有 {len(below)} 类原文 F1 低于 90%。'
              '应逐类检查标签噪声、语句本身的可判别性和训练覆盖，不能用总体平均数替代逐语言验收。'
              '未覆盖外部测试的类别也不能声称达标。', '',
              '这仍是原型评测：语料标签没有逐条人工审校，短句可能包含人名、借词、混合语言和旧编码；'
              '40 字符测试属于派生版本。尚未测真实业务流量、未知语言拒识、混合语言路由或置信度校准。'
              '基座预训练可能接触过相关公开语料，不能承诺完全无预训练污染。', '',
              '错误抽查确认测试来源也有错标：例如白俄罗斯语标签下出现英文 “Interval start time”，'
              '俄语标签下出现英文论坛时间模板。因此本表是按原始来源标签计算的分数，'
              '不能将每一处与来源标签不一致都解释为模型识别错误；本次没有用模型预测重标测试集。', '',
              '保存后的增强适配器已单独重新加载；冰岛语、英语、中文、西班牙语和丹麦语五个手工样例均得到正确首选标签。'
              '这是产物可用性检查，不是正式准确率估计；详细输出见 `manual_predictions.jsonl`。', '',
              '下一步优先补人工核验的冰岛语短句与近缘语言对照，建立新的业务保留集；'
              '优先做质量检查和针对性补样，而不是无差别继续堆量。', '',
              '## 复现与产物', '',
              '```bash',
              '.venv/bin/python scripts/predict.py --run runs/expanded_lora_109 "Ég tala íslensku á hverjum degi."',
              '```', '',
              '- `runs/expanded_lora_109/best_adapter/`：增强版适配器与分类头。',
              '- `runs/expanded_baseline/`：原模型在同一新测试集上的结果。',
              '- `runs/expanded_lora_109/per_language.csv`：每类原文和短文本指标，外部未测类别留空。',
              '- `data/expanded/summary.json` 与三个 JSONL：实际数据量、来源、父样本和指纹。',
              '- `data/opus_ranges/`：固定提交的范围缓存；`requirements.lock.txt`：本机依赖。', '',
              '数据来源：[OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)、'
              '[WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018)。'
              'OPUS-100 数据集卡许可证标为 unknown，底层语料条款各异；本次作本地实验，不重新分发语料。'
              'WiLI 数据集卡标为 ODbL。', '']
    report = '\n'.join(lines)
    (ROOT / 'local_evaluation_report.md').write_text(report, encoding='utf-8')
    print(report)

if __name__ == '__main__':
    main()
