
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


# ============== 工具函数 ==============
def create_dataloader(dataset, batch_size, shuffle):
    """统一构造 DataLoader，复用 collate_fn。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip_norm):
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
def evaluate(model, loader, device, id2label):
    """在给定 loader 上评估，返回 loss + 实体级 P/R/F1。"""
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


# ============== 主训练函数 ==============
def train_model(cfg):
    # ---- 1. 基础环境 ----
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ---- 2. 标签 + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_name)

    # ---- 3. 数据集 & DataLoader ----
    train_ds = NerDataset(
        data_path=cfg.data.train_path,
        class_path=cfg.data.class_path,
        tokenizer=tokenizer,
        max_len=cfg.train.max_len,
    )
    dev_ds = NerDataset(
        data_path=cfg.data.dev_path,
        class_path=cfg.data.class_path,
        tokenizer=tokenizer,
        max_len=cfg.train.max_len,
    )
    test_ds = NerDataset(
        data_path=cfg.data.test_path,
        class_path=cfg.data.class_path,
        tokenizer=tokenizer,
        max_len=cfg.train.max_len,
    )

    train_loader = create_dataloader(train_ds, cfg.train.batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_ds, cfg.train.batch_size, shuffle=False)
    test_loader = create_dataloader(test_ds, cfg.train.batch_size, shuffle=False)

    print(f"训练集：{len(train_ds)} | 验证集：{len(dev_ds)} | 测试集：{len(test_ds)}")
    id2label = train_ds.id2label

    # ---- 4. SwanLab 初始化（online 模式，不会触发 swanboard 报错）----
    swanlab.init(
        project="bert-ner-demo2",
        experiment_name=cfg.project.experiment_name,
        config={
            "pretrain": cfg.model.pretrain_name,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "epoch": cfg.train.epoch,
            "max_len": cfg.train.max_len,
            "seed": cfg.train.seed,
            "warmup_rate": cfg.train.warmup_rate,
            "dropout": cfg.train.dropout,
            "num_labels": len(id2label),
        },
        mode="online",
    )

    # ---- 5. 模型 / 优化器 / 学习率调度 ----
    model = BertNERModel(
        pretrain_name=cfg.model.pretrain_name,
        num_labels=len(id2label),
        dropout=cfg.train.dropout
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)
    total_steps = len(train_loader) * cfg.train.epoch
    warmup_steps = int(total_steps * cfg.train.warmup_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- 6. 训练循环（带早停 + 保存最优模型）----
    best_f1 = 0.0
    patience = cfg.train.early_stop_patience
    stop_cnt = 0
    best_model_path = os.path.join(cfg.save.model_dir, "best_model.pt")
    os.makedirs(cfg.save.model_dir, exist_ok=True)

    for epoch in range(cfg.train.epoch):
        print(f"\n{'='*60}\nEpoch {epoch + 1}/{cfg.train.epoch}\n{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, cfg.train.grad_clip_norm,
        )
        dev_result = evaluate(model, dev_loader, device, id2label)

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

        if dev_result["f1"] >= best_f1:
            best_f1 = dev_result["f1"]
            stop_cnt = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最优模型 | best F1 = {best_f1:.4f}")
        else:
            stop_cnt += 1
            print(f"F1 未提升，连续 {stop_cnt} / {patience} 轮")
            if stop_cnt >= patience:
                print("🛑 触发早停，训练结束")
                break

    swanlab.log({"best/dev_f1": best_f1})

    # ---- 7. 训练结束，加载最优模型，在测试集上做最终评估 ----
    print(f"\n{'='*60}\n加载最优模型，在测试集上评估\n{'='*60}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_result = evaluate(model, test_loader, device, id2label)

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

    swanlab.finish()
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
    train_model(cfg)
