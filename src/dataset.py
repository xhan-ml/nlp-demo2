import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NerDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        class_path: str | None,
        tokenizer: BertTokenizer,
        max_len: int = 128
    ):
        self.data_path = data_path
        self.class_path = class_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        
        if class_path is not None:
            entity_types = []
            with open(class_path, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        entity_types.append(t)
            label_list = []
            for et in entity_types:
                label_list.append(f"B-{et}")
                label_list.append(f"I-{et}")
            label_list.append("O")
        else:
   
            label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

        self.samples = self._load_data(data_path)

    
    def _load_data(self, file_path: str):
        samples = []
        tokens = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
               
                if not line:
                    if tokens:
                        
                        samples.append({
                            "tokens": tokens,
                            "labels": labels
                        })
                        tokens = []
                        labels = []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                char = parts[0]
                tag = parts[1]
                tokens.append(char)
                labels.append(tag)
        
            if tokens:
               samples.append({
                "tokens": tokens,
                "labels": labels
                })

        return samples

    def _tokenize_and_align_labels(self, tokens, labels):
        # padding=False，不预先填充，交给collate_fn批量动态padding
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True
        )
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
           
            if word_id is None:
                aligned_labels.append(-100)
            # 原始单词的第一个subword，赋予真实标签
            elif word_id != previous_word_id:
                label = labels[word_id]


                
                aligned_labels.append(self.label2id[label])
            else:
                aligned_labels.append(-100)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": aligned_labels
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        encoded = self._tokenize_and_align_labels(
            sample["tokens"],
            sample["labels"]
        )
        return encoded

    def collate_fn(self, batch):
        # 取出input_ids、attention_mask交给tokenizer.pad批量填充
        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            }
            for item in batch
        ]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        batch_max_seq_len = padded["input_ids"].size(1)

        # 对labels做padding，padding位置全部填充-100
        padded_label_list = []
        for item in batch:
            label_seq = item["labels"]
            pad_len = batch_max_seq_len - len(label_seq)
            label_seq = label_seq + [-100] * pad_len
            padded_label_list.append(label_seq)

        padded["labels"] = torch.tensor(padded_label_list, dtype=torch.long)
        return padded

