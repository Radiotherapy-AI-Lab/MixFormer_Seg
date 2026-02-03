import os
import json
import logging
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入你自定义的模块
from modelt import HeadNeck_Swin_Res2Net
from losses.loss import Loss
from optimizers.lr_scheduler import WarmupCosineSchedule
from data_loader import get_loader_data_622

# --- 配置区 (代替命令行参数) ---
class Config:
    logdir = "./runs/test_experiment"
    epochs = 100
    batch_size = 2
    lr = 4e-5
    decay = 0.001
    eval_num = 1
    warmup_epochs = 5
    early_stop_patience = 10
    # 数据路径
    csv_path = r"mask_DS_stratified_processed.csv"
    json_path = r"medical_cases_en.json"
    load_split_path = r"train_val_test_data_list.csv"

# --- 辅助函数 ---
def compute_dice(pred, target, epsilon=1e-7):
    """计算Dice系数"""
    # 自动处理 Sigmoid 和 二值化
    if not (pred.min() >= 0 and pred.max() <= 1):
        pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    
    intersection = (pred_bin * target).sum(dim=(1, 2, 3, 4))
    union = pred_bin.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return ((2. * intersection) / (union + epsilon)).mean().item()

def save_ckp(state, path, logger):
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")

# --- 核心逻辑 ---
def train_one_epoch(model, loader, optimizer, scheduler, loss_func, epoch, logger):
    model.train()
    total_loss, total_dice = 0, 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    
    for batch in pbar:
        optimizer.zero_grad()
        # 准备数据并调整维度 [B, C, H, W, D]
        x = batch["image"].cuda().permute(0, 1, 4, 2, 3).contiguous()
        mask = batch["mask"].cuda().permute(0, 1, 4, 2, 3).contiguous()
        text = batch["text"]

        pred, extra_loss = model(x, text, mask)
        loss = loss_func(pred, mask) + extra_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        dice = compute_dice(pred.detach(), mask)
        total_loss += loss.item()
        total_dice += dice
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})
        
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, loss_func, epoch, logger, desc="Val"):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{desc} Epoch {epoch+1}"):
            x = batch["image"].cuda().permute(0, 1, 4, 2, 3).contiguous()
            mask = batch["mask"].cuda().permute(0, 1, 4, 2, 3).contiguous()
            text = batch["text"]

            pred, extra_loss = model(x, text)
            loss = loss_func(pred, mask) + extra_loss
            
            total_loss += loss.item()
            total_dice += compute_dice(pred, mask)
            
    avg_loss, avg_dice = total_loss / len(loader), total_dice / len(loader)
    logger.info(f"{desc} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

# --- 主程序 ---
def main():
    cfg = Config()
    os.makedirs(cfg.logdir, exist_ok=True)
    
    # 日志与可视化
    writer = SummaryWriter(cfg.logdir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # 模型、损失、优化器
    model = HeadNeck_Swin_Res2Net(cfg).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)
    loss_func = Loss()

    # 数据加载
    train_loader, val_loader, test_loader = get_loader_data_622(
        cfg, csv_path=cfg.csv_path, json_path=cfg.json_path, load_split_path=cfg.load_split_path
    )

    # 学习率调度器
    total_steps = len(train_loader) * cfg.epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_epochs, t_total=total_steps)

    # 训练循环
    best_dice = 0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        t_loss, t_dice = train_one_epoch(model, train_loader, optimizer, scheduler, loss_func, epoch, logger)
        writer.add_scalar("train/loss", t_loss, epoch)
        writer.add_scalar("train/dice", t_dice, epoch)

        # 验证
        if (epoch + 1) % cfg.eval_num == 0:
            v_loss, v_dice = validate(model, val_loader, loss_func, epoch, logger)
            writer.add_scalar("val/loss", v_loss, epoch)
            writer.add_scalar("val/dice", v_dice, epoch)

            # 保存最佳并检查早停
            if v_dice > best_dice:
                best_dice = v_dice
                patience_counter = 0
                save_ckp({"state_dict": model.state_dict(), "dice": v_dice}, 
                         os.path.join(cfg.logdir, "best_model.pth"), logger)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

        # 定期测试
        if (epoch + 1) % 5 == 0:
            validate(model, test_loader, loss_func, epoch, logger, desc="Test")

    writer.close()

if __name__ == "__main__":
    main()
