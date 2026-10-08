import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import BertTokenizer
from src.config import ConfigManager
from src.model import BertNERModel
from src.utils import find_latest_model, NEREntityMetric


def main():
    # 1. 加载配置
    cfg = ConfigManager.load_from_json("checkpoints/test/experiment_config.json")

    # 2. 加载分词器
    tokenizer = BertTokenizer.from_pretrained(cfg.model.pretrain_name)

    # 3. 构造 id2label（和 dataset 保持一致）
    if cfg.data.class_path is not None:
        with open(cfg.data.class_path, "r", encoding="utf-8") as f:
            label_list = [line.strip() for line in f if line.strip()]
    else:
        label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    id2label = {i: lab for i, lab in enumerate(label_list)}
    num_labels = len(label_list)

    # 4. 初始化模型 + 加载最优权重
    model = BertNERModel(pretrain_name=cfg.model.pretrain_name, num_labels=num_labels)
    model_path = find_latest_model(model_dir="checkpoint")  # 你的实验保存根目录
    print(f"加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 5. 输入句子
    text = "张三在北京上学，李四去上海出差。"
    chars = list(text)
    print(f"\n输入句子: {text}")

    # 6. 编码（和 dataset 里的方式一致）
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=cfg.train.max_len,
        padding=False,
        return_attention_mask=True,
    )
    input_ids = torch.tensor([encoding["input_ids"]])
    attention_mask = torch.tensor([encoding["attention_mask"]])

    # 7. 前向预测（不传 labels）
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    # 8. 去掉 [CLS]/[SEP]，按 word_ids 对齐回原始字符
    word_ids = encoding.word_ids()
    char_pred_ids = []
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue  # CLS / SEP
        char_pred_ids.append(pid)

    # 9. 解码实体
    pred_ents = NEREntityMetric.extract_entities(char_pred_ids, id2label)

    # 10. 打印结果
    print("\n识别结果：")
    for ent_type, start, end in sorted(pred_ents, key=lambda x: x[1]):
        entity_text = "".join(chars[start:end + 1])
        print(f"  [{ent_type}] {entity_text}  (位置 {start}-{end})")


if __name__ == "__main__":
    main()
