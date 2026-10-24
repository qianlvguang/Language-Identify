"""面向说明书 OCR 单条识别的生产级推理工具 (predict_ocr.py)。

功能特性：
1. 单条/批量输入即时语言识别；
2. 输出预测类别、中文名称与 Softmax 校准置信度；
3. 支持 --format online_38 选项，自动将宏语言兼容映射为线上 38 业务标签；
4. 内置低置信度未知文本/严重乱码安全过滤提示。
"""
import argparse
import json
import os
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.language_taxonomy import (
    TARGET_LANGUAGES, LANGUAGE_NAMES_87, to_online_format, is_core_38, resolve_hierarchical_fallback
)
from scripts.jev_engine import JevDecisionEngine

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def main():
    parser = argparse.ArgumentParser(description="OCR 说明书单条文本语言识别推理工具")
    parser.add_argument('text', nargs='?', type=str, help="待检测的单条文本")
    parser.add_argument('--adapter-path', type=Path, default=ROOT / 'runs/ocr_v2_model/best_adapter', help="模型适配器路径")
    parser.add_argument('--base-model-path', type=Path, default=ROOT / 'models/xlm_roberta_base', help="基础模型路径")
    parser.add_argument('--format', choices=['target_87', 'online_38'], default='online_38', help="输出标签格式")
    parser.add_argument('--confidence-threshold', type=float, default=0.45, help="置信度安全阈值")
    parser.add_argument('--use-jev', action='store_true', help="启用 Jev 结构化决策引擎 (执行 Noul 纯规格过滤与 Choice 校准)")
    args = parser.parse_args()

    clean_text = unicodedata.normalize('NFC', args.text).strip()

    # 1. 若开启 Jev 引擎，首先运行 Noul 原语：判断是否为纯电气规格/条码噪声
    jev = JevDecisionEngine()
    if args.use_jev:
        is_natural, jev_conf = jev.noul_is_valid_natural_language(clean_text)
        if not is_natural:
            result = {
                'input_text': clean_text,
                'predicted_code': 'non_language_spec',
                'language_name': '纯电气规格/参数符号（非自然语言）',
                'confidence': jev_conf,
                'decision_tier': 'jev_noul_filtered',
                'is_core_38': False,
                'is_low_confidence': False,
                'note': 'Jev.Noul 成功过滤说明书纯度量衡/电气参数杂质'
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

    if not args.text:
        # 支持 stdin 管道输入
        if not sys.stdin.isatty():
            args.text = sys.stdin.read().strip()
        else:
            print("请输入待识别文本，例如: python scripts/predict_ocr.py 'Mode d\\'emploi'")
            sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    adapter_p = args.adapter_path
    if not adapter_p.exists():
        fallback = ROOT / 'runs/expanded_lora_109/best_adapter'
        if fallback.exists():
            print(f"[提示] 指定的 {adapter_p} 尚未生成，自动临时使用已有模型 {fallback} 进行演示推理。")
            adapter_p = fallback
        else:
            raise FileNotFoundError(f"未找到可用模型权重: {adapter_p}")

    # 优先从适配器自身读取标签列表，保证向下与跨版本完全兼容
    adapter_langs_p = adapter_p / 'languages.json'
    parent_langs_p = adapter_p.parent / 'languages.json'
    if adapter_langs_p.exists():
        labels = json.loads(adapter_langs_p.read_text(encoding='utf-8'))
    elif parent_langs_p.exists():
        labels = json.loads(parent_langs_p.read_text(encoding='utf-8'))
    else:
        labels = sorted(TARGET_LANGUAGES)

    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(adapter_p, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
        attn_implementation='eager'
    )
    model = PeftModel.from_pretrained(base_model, adapter_p)
    model.to(device)
    model.eval()

    clean_text = unicodedata.normalize('NFC', args.text).strip()
    encoded = tokenizer([clean_text], padding=True, truncation=True, max_length=96, return_tensors='pt')

    with torch.inference_mode():
        logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
        probs = logits.softmax(dim=-1).cpu()[0]
        top_prob, top_idx = torch.topk(probs, 3)

    # 构造全量概率字典供谱系降级解析
    probs_dict = {id2label[i]: float(p) for i, p in enumerate(probs.tolist())}
    fallback_info = resolve_hierarchical_fallback(probs_dict, threshold=args.confidence_threshold)

    pred_code = id2label[int(top_idx[0])]
    confidence = float(top_prob[0])

    # 格式转换
    final_code = to_online_format(fallback_info['resolved_code'], prefer_legacy=True) if args.format == 'online_38' else fallback_info['resolved_code']
    lang_name = LANGUAGE_NAMES_87.get(final_code, LANGUAGE_NAMES_87.get(pred_code, pred_code))

    result = {
        'input_text': clean_text,
        'predicted_code': final_code,
        'language_name': lang_name,
        'confidence': round(confidence, 4),
        'decision_tier': fallback_info.get('tier', 'leaf'),
        'hierarchical_family': fallback_info.get('family_name'),
        'family_aggregated_confidence': fallback_info.get('confidence') if fallback_info.get('tier') == 'family_fallback' else None,
        'is_core_38': is_core_38(final_code),
        'is_low_confidence': confidence < args.confidence_threshold and fallback_info.get('tier') != 'family_fallback',
        'top_candidates': [
            {'code': id2label[int(idx)], 'name': LANGUAGE_NAMES_87.get(id2label[int(idx)], id2label[int(idx)]), 'prob': round(float(p), 4)}
            for p, idx in zip(top_prob, top_idx)
        ]
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
