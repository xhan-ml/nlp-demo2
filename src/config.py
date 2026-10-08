import os
import json
import argparse
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    project: str
    experiment_name: str

@dataclass
class ModelConfig:
    pretrain_name: str

@dataclass
class DataConfig:
    train_path: str
    dev_path: str
    test_path: str
    class_path: str | None  

@dataclass
class TrainConfig:
    seed: int
    epoch: int
    batch_size: int
    max_len: int
    lr: float
    warmup_rate: float
    early_stop_patience: int
    dropout: float
    grad_clip_norm: float

@dataclass
class SaveConfig:
    model_dir: str
    log_dir: str


class ConfigManager:
    def __init__(self):
        self.raw_config = None

    @staticmethod
    def get_args():
         
        parser = argparse.ArgumentParser()
     
        parser.add_argument("--exp_name", type=str, required=True, help="实验名，例：01_msra_bert_base")
        parser.add_argument("--dataset", type=str, required=True, choices=["msra", "weibo"])
        parser.add_argument("--pretrain_name", type=str, required=True)

        # 训练超参
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--epoch", type=int, default=10)
       
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--max_len", type=int, default=128)
        parser.add_argument("--lr", type=float, default=2e-5)
        parser.add_argument("--warmup_rate", type=float, default=0.1)
        parser.add_argument("--early_stop_patience", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--grad_clip_norm", type=float, default=1.0)
        args = parser.parse_args()
        return args

    def build_and_save(self, args):
       
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
      
        ckpt_root = os.path.join(BASE_DIR, "checkpoints")
        model_dir = os.path.join(ckpt_root, args.exp_name)
        log_dir = os.path.join(BASE_DIR, "logs", args.exp_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

       
        if args.dataset == "msra":
            
            data_base = os.path.join(BASE_DIR, "data", "MSRA")
            train_path = os.path.join(data_base, "train_5k.txt")
            dev_path = os.path.join(data_base, "dev_1k.txt")
            test_path = os.path.join(data_base, "test_1k.txt")
            class_path = None
           
        elif args.dataset == "weibo":
            data_base = os.path.join(BASE_DIR, "data", "weibo")
            train_path = os.path.join(data_base, "train.txt")
            dev_path = os.path.join(data_base, "dev.txt")
            test_path = os.path.join(data_base, "test.txt")
            class_path = os.path.join(data_base, "class.txt")
         
      
        cfg_dict = {
            "project": {
                "project": "mini_bert_ner",
                "experiment_name": args.exp_name
            },
            "model": {
                "pretrain_name": args.pretrain_name
            },
            "data": {
                "train_path": train_path,
                "dev_path": dev_path,
                "test_path": test_path,
                "class_path": class_path
            },
            "train": {
                "seed": args.seed,
                "epoch": args.epoch,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "lr": args.lr,
                "warmup_rate": args.warmup_rate,
                "early_stop_patience": args.early_stop_patience,
                "dropout": args.dropout,
                "grad_clip_norm": args.grad_clip_norm
            },
            "save": {
                "model_dir": model_dir,
                "log_dir": log_dir
            }
        }
       
        json_save_path = os.path.join(model_dir, "experiment_config.json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        self.raw_config = cfg_dict
        return json_save_path

    @staticmethod
    def load_from_json(json_path: str):

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        project = ProjectConfig(**raw["project"])
        model = ModelConfig(**raw["model"])
        data = DataConfig(**raw["data"])
        train = TrainConfig(**raw["train"])
        save = SaveConfig(**raw["save"])
     
        class ConfigWrap:
            def __init__(self, p, m, d, t, s):
                self.project = p
                self.model = m
                self.data = d
                self.train = t
                self.save = s
        return ConfigWrap(project, model, data, train, save)


if __name__ == "__main__":

    manager = ConfigManager()
    args = ConfigManager.get_args()
    json_path = manager.build_and_save(args)
    cfg = ConfigManager.load_from_json(json_path)

    print("配置加载成功！")
    print(f"实验名称：{cfg.project.experiment_name}")
    print(f"学习率：{cfg.train.lr}")
    print(f"训练集路径：{cfg.data.train_path}")
