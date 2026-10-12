
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import swanlab
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


from src.config import ConfigManager
from src.dataset import NerDataset
from src.model import BertNERModel
from src.utils import set_seed, NEREntityMetric


class NerTrainer:
    def __init__(self,cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.train_ds = None
        self.dev_ds = None
        self.test_ds = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.id2label = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def create_dataloader(self,dataset, batch_size, shuffle):
        """统一构造 DataLoader，复用 collate_fn。"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
        )


    def train_one_epoch(self, model, loader, optimizer, scheduler, device, grad_clip_norm):
        """训练一个 epoch，返回平均训练 loss。"""
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
    
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    
            optimizer.step()
            scheduler.step()
    
            total_loss += loss.item()
        return total_loss / len(loader)


    @torch.no_grad()
    def evaluate(self,model, loader, device, id2label):
        """返回loss，以及三个指标p。r，f。"""
        model.eval()
        total_loss = 0.0
        metric = NEREntityMetric(id2label)
    
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
    
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            metric.update(
                true_labels_list=labels.cpu().tolist(),
                pred_labels_list=pred.cpu().tolist(),
            )
        result = metric.compute()
        result["loss"] = total_loss / len(loader)
        return result

    def run(self):
        # ---- 1. 基础环境 ----
        set_seed(self.cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备：{device}")
        #2.加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrain_name)
        
        print("tokenizer对象：", self.tokenizer)
        #3.构建数据集
        self.train_ds = NerDataset(
            data_path=self.cfg.train_path,
            class_path=self.cfg.class_path,
            tokenizer=self.tokenizer,
            max_len=self.cfg.max_len,
        )
        self.dev_ds = NerDataset(
            data_path=self.cfg.dev_path,
            class_path=self.cfg.class_path,
            tokenizer=self.tokenizer,
            max_len=self.cfg.max_len,
        )
        self.test_ds = NerDataset(
            data_path=self.cfg.test_path,
            class_path=self.cfg.class_path,
            tokenizer=self.tokenizer,
            max_len=self.cfg.max_len,
        )
        # 把数据集打包成dataloader，按批次喂给模型
        self.train_loader = self.create_dataloader(self.train_ds, self.cfg.batch_size, shuffle=True)
        self.dev_loader = self.create_dataloader(self.dev_ds, self.cfg.batch_size, shuffle=False)
        self.test_loader = self.create_dataloader(self.test_ds, self.cfg.batch_size, shuffle=False)
        print(f"训练集：{len(self.train_ds)} | 验证集：{len(self.dev_ds)} | 测试集：{len(self.test_ds)}")
        
        self.id2label = self.train_ds.id2label
        # 4.swanlab可视化，将loss，p，r，f记录起来
        swanlab.init(
            project="bert-ner-demo2",
            experiment_name=self.cfg.experiment_name,
            config={
                "pretrain": self.cfg.pretrain_name,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
                "epoch": self.cfg.epoch,
                "max_len": self.cfg.max_len,
                "seed": self.cfg.seed,
                "warmup_rate":self.cfg.warmup_rate,
                "dropout": self.cfg.dropout,
                "num_labels": len(self.id2label),
            },
            mode="online",
        )
        # 5.
        self.model = BertNERModel(
            pretrain_name=self.cfg.pretrain_name,
            num_labels=len(self.id2label),
            dropout=self.cfg.dropout
        )
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr)
        total_steps = len(self.train_loader) * self.cfg.epoch
        warmup_steps = int(total_steps * self.cfg.warmup_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        #6.训练循环（带早停 + 保存最优模型）
        best_f1 = 0.0
        patience = self.cfg.early_stop_patience  #连续多少轮不提升就停止
        stop_cnt = 0 #记录f没有提升的轮数
        best_model_path = os.path.join(cfg.model_dir, "best_model.pt")
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        for epoch in range(self.cfg.epoch):
            print(f"\n{'='*60}\nEpoch {epoch + 1}/{self.cfg.epoch}\n{'='*60}")
            train_loss = self.train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.scheduler,
                self.device, self.cfg.grad_clip_norm,
            )
            dev_result = self.evaluate(self.model, self.dev_loader, self.device, self.id2label)
    
            print(f"Train Loss: {train_loss:.4f} | Dev Loss: {dev_result['loss']:.4f}")
            print(f"Dev P: {dev_result['precision']:.4f}  "
                  f"R: {dev_result['recall']:.4f}  "
                  f"F1: {dev_result['f1']:.4f}")
            swanlab.log({
                "train/loss": train_loss,
                "dev/loss": dev_result["loss"],
                "dev/precision": dev_result["precision"],
                "dev/recall": dev_result["recall"],
                "dev/f1": dev_result["f1"],
            }, step=epoch + 1)
            #判断当前是不是最优模型
            if dev_result["f1"] >= best_f1:
                best_f1 = dev_result["f1"]
                stop_cnt = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"✅保存最优模型 | best F1 = {best_f1:.4f}")
            else:
                stop_cnt += 1
                print(f"F1 未提升，连续 {stop_cnt} / {patience} 轮")
                if stop_cnt >= patience:
                    print("🛑触发早停，训练结束")
                    break
        swanlab.log({"best/dev_f1": best_f1})
        # 7.训练结束，加载最优模型，评估
        print(f"\n{'='*60}\n加载最优模型，在测试集上评估\n{'='*60}")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        test_result = self.evaluate(self.model, self.test_loader, self.device, self.id2label)
    
        print(f"Test Loss: {test_result['loss']:.4f}")
        print(f"Test P: {test_result['precision']:.4f}  "
              f"R: {test_result['recall']:.4f}  "
              f"F1: {test_result['f1']:.4f}")
        swanlab.log({
            "test/loss": test_result["loss"],
            "test/precision": test_result["precision"],
            "test/recall": test_result["recall"],
            "test/f1": test_result["f1"],
        })
        swanlab.finish()      #结束swanlab评估
        print(f"\n全部实验完成 | Best Dev F1: {best_f1:.4f} | Test F1: {test_result['f1']:.4f}")
        return {
            "best_dev_f1": best_f1,
            "test_metrics": test_result,
        }


# ============== 入口 ==============
if __name__ == "__main__":
    manager = ConfigManager()
    args = ConfigManager.get_args()
    json_path = manager.build_and_save(args)
    cfg = ConfigManager.load_from_json(json_path)
    trainer=NerTrainer(cfg)
    trainer.run()
