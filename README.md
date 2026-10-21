# Language-Identify

基于 `FacebookAI/xlm-roberta-base` 的 **109 类语言识别**项目，包含冰岛语。使用 XLM-R + LoRA + 分类头，提供数据准备、清洗、训练、评估与推理脚本。

**已实测：M2 Max、30 核 GPU、32GB 统一内存，Python 3.14.3、PyTorch MPS。** 当前训练脚本明确要求 MPS，未验证 CUDA、全参数训练或百万条数据规模。

## 结果与覆盖

见 [109 类完整报告](reports/clean_round_report.md)、[逐语言指标](reports/per_language.csv)、[数值汇总](reports/latest_metrics.json)。只能比较同一测试集上的前后分数，不能把之前 88 类 OPUS 成绩与新测试成绩相减。

| 同一测试集指标 | 上一版 | 清洗后追加一轮 |
|---|---:|---:|
| 原文 Macro-F1 | 98.08% | 98.06% |
| 前40字符 Macro-F1（去除输入重合） | 93.53% | 93.88% |
| 冰岛语原文 F1 | 98.52% | 98.04% |
| 冰岛语前40字符 F1 | 85.05% | 83.65% |

这轮不是全面提升。**默认保留上一版 `expanded_lora_109`**；本轮 `clean_lora_109` 作为对照候选。部分近缘语言仍明显偏低，请查看逐语言指标，不能仅凭均分宣称全语种达标。

- 新原文测试 **109 类各 100 条，共 10,900 条**，此前缺失的 21 类已全部补齐。
- 106 类来自固定版本 FLORES devtest；布列塔尼语、西弗里斯兰语、拉丁语来自独立 UD 测试文本。
- 完整测试输入与训练/验证无重合；前 40 字符派生测试另报告输入去重后的结果。
- 标签采用 ISO 639-3：英语 `eng`、中文 `zho`、冰岛语 `isl`。中文不分简繁，挪威语区分 `nob` 和 `nno`。
- 见 [语言清单](languages.csv) 和 [逐类来源与变体](reports/test_sources.csv)。一个标准变体通过不等于所有方言、转写和书写系统均通过。

这是原型评估：尚未测试真实业务分布、未知/混合语言拒识，softmax 分数未校准。公开语料可能有标注歧义、借词或近重复；每类 100 条测试也不足以承诺稳定业务错误率。

## 安装

```bash
git clone https://github.com/qianlvguang/Language-Identify.git
cd Language-Identify
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.lock.txt
.venv/bin/python scripts/download_assets.py
```

需事先安装 uv；FastText 从源码构建需要 C++ 编译工具。基座下载约 1.1GB。模型、数据和范围缓存保存在项目内，不纳入版本控制。

## 从零复现

先生成父适配器，再准备扩充与清洗数据。仓库不分发原始语料或权重。

```bash
# 百科小样本基线：109 类、3 epoch
.venv/bin/python scripts/local_experiment.py --output runs/local_lora_109

# 扩充真实短句与训练增强，继续训练 2 epoch
.venv/bin/python scripts/prepare_expanded.py
.venv/bin/python scripts/local_experiment.py --prepared-data data/expanded --init-adapter runs/local_lora_109/best_adapter --epochs 2 --output runs/expanded_lora_109

# 清洗并建立全新 109 类测试，独立审计输入与父样本隔离
.venv/bin/python scripts/prepare_clean_round.py
.venv/bin/python scripts/audit_expanded.py --data-dir data/clean_round

# 在同一新测试上评测上一版
.venv/bin/python scripts/local_experiment.py --prepared-data data/clean_round --init-adapter runs/expanded_lora_109/best_adapter --evaluate-only --output runs/clean_baseline

# 清洗后追加一轮并评测、生成报告
.venv/bin/python scripts/local_experiment.py --prepared-data data/clean_round --init-adapter runs/expanded_lora_109/best_adapter --epochs 1 --output runs/clean_lora_109
.venv/bin/python scripts/write_clean_report.py
```

参数：LoRA r=8、alpha=16、dropout=0.05，作用于 query/value；分类头可训练。FP32、batch 32、96 token，AdamW，适配器学习率 `2e-4`、分类头 `1e-3`、6% warmup。扩充/清洗阶段按语言逆频率有放回采样。随机种子、数据和父适配器指纹记录在运行产物中。

验证集选择适配器；测试不进入训练，不按教师或本模型预测改写答案。新测试排除历史实验已见文本与相同完整模型输入，派生短文本重合另行剔除。

## 清洗与合成

- OPUS-100 按固定提交读取少量分散行组，保留范围缓存，不依赖网页预览 API。
- 独立 FastText 辅助排除高置信度、非近缘标签冲突；保留不确定项与近缘语言来源标签，不把教师当成真值。
- 清理复制英文的外语侧、明显文字脚本不符、论坛时间模板，同步删除失去真实父样本的衍生项。
- 增强方式为训练父文本的随机短片段、标点/数字扰动及冰岛语旧字幕编码变体；保留 source、parent、synthetic 和哈希，不用于测试。
- 本轮训练 **59,675 → 57,614 条**，验证 **5,000 → 4,750 条**，训练保留全部 109 类。

审计见 [data_audit.json](reports/data_audit.json)，实验溯源见 [experiment_manifest.json](reports/experiment_manifest.json)。

## 推理

```bash
.venv/bin/python scripts/predict.py 'Ég tala íslensku á hverjum degi.'
# 显式加载本轮对照候选
.venv/bin/python scripts/predict.py --run runs/clean_lora_109 'Ég tala íslensku á hverjum degi.'
.venv/bin/python scripts/predict.py --run runs/clean_lora_109 '这是一段中文。' 'This is an English sentence.'
```

需要本地基座与运行目录中的 languages.json、config.json、best_adapter/。输出前三个标签及未经校准的模型分数，不承诺极短、纯数字或混合语言输入能唯一判定。

## 发布检查

```bash
python3 -m compileall -q scripts
python3 scripts/validate_release.py
```

GitHub Actions 只检查语法与发布产物，不等于运行了 GPU 训练。真实训练证据为报告中的实测结果、时间、数据隔离审计和实验指纹。

## 来源与许可证

代码为 [MIT](LICENSE)，不覆盖第三方模型、语料或权重。

- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)：固定版本基座。
- [WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018)：数据集卡 ODbL；段落可能与文章语言标签不符。
- [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)：数据集卡 unknown，底层语料条款各异；不重新分发。
- [FLORES](https://huggingface.co/datasets/mteb/flores)：CC BY-SA 4.0，本实验仅用 devtest 测试。
- [UD Latin LLCT](https://github.com/UniversalDependencies/UD_Latin-LLCT)、[UD Breton KEB](https://github.com/UniversalDependencies/UD_Breton-KEB)、[UD Frisian Frysk](https://github.com/UniversalDependencies/UD_Frisian-Frysk/tree/dev)：本次采用的测试文本为 CC BY-SA 4.0。
- [FastText](https://fasttext.cc/docs/en/language-identification.html)：仅辅助训练数据质量筛查。

仅发布代码、依赖、中文报告、来源和聚合数值。不提交原始语料、缓存、虚拟环境、模型权重或本机私有配置。
