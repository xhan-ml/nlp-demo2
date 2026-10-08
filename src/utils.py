"""
工具函数 + NER实体级指标
"""
import os
import random
from datetime import datetime
import numpy as np
import torch



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_model(model_dir: str):
   
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    candidates = []
    direct_path = os.path.join(model_dir, "best_model.pt")
    if os.path.isfile(direct_path):
        candidates.append(direct_path)
    for entry in os.scandir(model_dir):
        if entry.is_dir():
            model_path = os.path.join(entry.path, "best_model.pt")
            if os.path.isfile(model_path):
                candidates.append(model_path)
    if not candidates:
        raise FileNotFoundError(f"在目录中未找到 best_model.pt: {model_dir}")
    return max(candidates, key=os.path.getmtime)


class NEREntityMetric:
    @staticmethod
    def extract_entities(label_sequence: list, id2label: dict):
       
        entities = set()
        current_type = None
        start = None
        def close_entity(end_index):
            nonlocal current_type, start
            if current_type is not None and start is not None:
                entities.add((current_type, start, end_index))
            current_type = None
            start = None
        for idx, label_id in enumerate(label_sequence):
            label = id2label[int(label_id)]
            if label.startswith("B-"):
                close_entity(idx - 1)
                current_type = label[2:]
                start = idx
            elif label.startswith("I-"):
                if current_type != label[2:]:
                    close_entity(idx - 1)
            else:
                close_entity(idx -1)
        close_entity(len(label_sequence)-1)
        return entities




    def __init__(self, id2label: dict):
        self.id2label = id2label
     
        self.reset()

    def reset(self):
        
        self.total_tp = 0
        self.total_pred_entities = 0
        self.total_true_entities = 0

    def update(self, true_labels_list: list, pred_labels_list: list):
    
        for true_ids, pred_ids in zip(true_labels_list, pred_labels_list):
            valid_pairs = [
                (true_id, pred_id)
                for true_id, pred_id in zip(true_ids, pred_ids)
                if true_id != -100
            ]
            if not valid_pairs:
                continue
            true_valid = [t for t, _ in valid_pairs]
            pred_valid = [p for _, p in valid_pairs]

            true_ents = self.extract_entities(true_valid, self.id2label)
            pred_ents = self.extract_entities(pred_valid, self.id2label)

            self.total_tp += len(true_ents & pred_ents)
            self.total_pred_entities += len(pred_ents)
            self.total_true_entities += len(true_ents)

    def compute(self) -> dict:
    
        precision = self.total_tp / self.total_pred_entities if self.total_pred_entities > 0 else 0.0
        recall = self.total_tp / self.total_true_entities if self.total_true_entities > 0 else 0.0
        denominator = self.total_pred_entities + self.total_true_entities
        f1 = 2 * self.total_tp / denominator if denominator > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
