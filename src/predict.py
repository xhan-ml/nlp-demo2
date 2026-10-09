import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.config import ConfigManager
from src.model import BertNERModel
from src.utils import find_latest_model, NEREntityMetric
from src.dataset import NerDataset
from transformers import AutoTokenizer



def main():
    # 1. 加载配置
    cfg = ConfigManager.load_from_json("checkpoints/weibo_bert_wwm_v3/experiment_config.json")

    print("class_path路径：", cfg.data.class_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_name) 

    train_ds = NerDataset(
        data_path=cfg.data.train_path,
        class_path=cfg.data.class_path,
        tokenizer=tokenizer,
        max_len=cfg.train.max_len
    )
  
    id2label = train_ds.id2label
    label_list = list(id2label.values())


        
    print("label_list = ", label_list)
    print("标签总数num_labels = ", len(label_list))


   

   
    num_labels = len(label_list)

    # 4. 初始化模型 + 加载最优权重
    model = BertNERModel(pretrain_name=cfg.model.pretrain_name, num_labels=num_labels)


    cfg_dir = os.path.dirname("checkpoints/weibo_bert_wwm_v3/experiment_config.json")
    model_path = find_latest_model(model_dir=cfg_dir)  
    
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


    print("char_pred_ids 标签id序列：", char_pred_ids)


    
    # 9. 解码实体
    pred_ents = NEREntityMetric.extract_entities(char_pred_ids, id2label)

    # 10. 打印结果
    print("\n识别结果：")
    for ent_type, start, end in sorted(pred_ents, key=lambda x: x[1]):
        entity_text = "".join(chars[start:end + 1])
        print(f"  [{ent_type}] {entity_text}  (位置 {start}-{end})")


if __name__ == "__main__":
    main()
