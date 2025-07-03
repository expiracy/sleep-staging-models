"""
训练改进的PPG+ECG模型 - 多GPU版本 (DDP)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, f1_score
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc
import argparse
import yaml
import warnings

warnings.filterwarnings('ignore')

from multimodal_model_crossattn import ImprovedMultiModalSleepNet
from multimodal_dataset_aligned import get_dataloaders


def setup(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()


class CrossAttentionTrainerDDP:
    def __init__(self, rank, world_size, config, run_id=None):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_id = run_id
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)

        # 获取模型类型
        self.model_type = config.get('model_type', 'generated_ecg')

        # 只在主进程创建目录和tensorboard
        if rank == 0:
            self.setup_directories()
            self.writer = SummaryWriter(self.log_dir)

            # 保存配置
            with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Using {world_size} GPUs for training")
            print(f"Model type: {self.model_type}")

        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()

    def setup_directories(self):
        """创建必要的目录（只在主进程执行）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"crossattn_{self.model_type}_v2_ddp_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"

        self.output_dir = os.path.join(self.config['output']['save_dir'], model_name)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.results_dir = os.path.join(self.output_dir, 'results')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_class_weights(self, train_dataset):
        """计算类别权重（只在主进程计算）"""
        if self.rank == 0:
            print("\nCalculating class weights...")

        all_labels = []
        sample_size = min(len(train_dataset), 50)

        # 确保所有进程使用相同的样本
        sample_indices = list(range(sample_size))

        for idx in sample_indices:
            _, _, labels = train_dataset[idx]
            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.numpy().tolist())

        label_counts = Counter(all_labels)
        class_counts = [label_counts.get(i, 1) for i in range(4)]
        total_samples = sum(class_counts)

        if self.rank == 0:
            print(f"\nLabel distribution:")
            stage_names = ['Wake', 'Light', 'Deep', 'REM']
            for i, count in enumerate(class_counts):
                percentage = count / total_samples * 100
                print(f"  {stage_names[i]}: {count} samples ({percentage:.2f}%)")

        class_weights = torch.tensor([total_samples / (4 * count) for count in class_counts],
                                     dtype=torch.float32)

        if self.rank == 0:
            print(f"\nClass weights: {class_weights}")

        return class_weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion, scheduler, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 设置epoch以确保不同epoch的数据打乱方式不同
        dataloader.sampler.set_epoch(epoch)

        # 只在主进程显示进度条
        if self.rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        for batch_idx, (ppg, ecg, labels) in enumerate(pbar):
            ppg = ppg.to(self.device)
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)

            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = model(ppg, ecg)
                    outputs_reshaped = outputs.permute(0, 2, 1)
                    loss = criterion(
                        outputs_reshaped.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(ppg, ecg)
                outputs_reshaped = outputs.permute(0, 2, 1)
                loss = criterion(
                    outputs_reshaped.reshape(-1, 4),
                    labels.reshape(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 更新学习率
            if scheduler is not None:
                scheduler.step()

            # 统计
            mask = labels != -1
            valid_outputs = outputs_reshaped[mask]
            valid_labels = labels[mask]

            if valid_labels.numel() > 0:
                _, predicted = valid_outputs.max(1)
                correct += predicted.eq(valid_labels).sum().item()
                total += valid_labels.numel()
                running_loss += loss.item() * valid_labels.numel()

            # 更新进度条（只在主进程）
            if self.rank == 0 and total > 0:
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': correct / total,
                    'lr': optimizer.param_groups[0]['lr']
                })

            # 定期清理内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # 同步所有进程的统计结果
        total_tensor = torch.tensor([total, correct, running_loss], device=self.device)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        total_all = total_tensor[0].item()
        correct_all = total_tensor[1].item()
        running_loss_all = total_tensor[2].item()

        epoch_loss = running_loss_all / total_all if total_all > 0 else 0
        epoch_acc = correct_all / total_all if total_all > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion):
        """验证（所有进程都参与）"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for ppg, ecg, labels in dataloader:
                ppg = ppg.to(self.device)
                ecg = ecg.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(ppg, ecg)
                        outputs_reshaped = outputs.permute(0, 2, 1)
                        loss = criterion(
                            outputs_reshaped.reshape(-1, 4),
                            labels.reshape(-1)
                        )
                else:
                    outputs = model(ppg, ecg)
                    outputs_reshaped = outputs.permute(0, 2, 1)
                    loss = criterion(
                        outputs_reshaped.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                mask = labels != -1
                valid_outputs = outputs_reshaped[mask]
                valid_labels = labels[mask]

                if valid_labels.numel() > 0:
                    running_loss += loss.item() * valid_labels.numel()
                    _, predicted = valid_outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(valid_labels.cpu().numpy())

        # 收集所有进程的预测结果
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 只在主进程计算指标
        if self.rank == 0:
            epoch_loss = running_loss / len(all_labels) if all_labels.size > 0 else 0
            accuracy = np.mean(all_preds == all_labels) if all_labels.size > 0 else 0
            kappa = cohen_kappa_score(all_labels, all_preds) if all_labels.size > 0 else 0
            f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels.size > 0 else 0

            cm = confusion_matrix(all_labels, all_preds)
            per_class_metrics = self.calculate_per_class_metrics(cm)

            return epoch_loss, accuracy, kappa, f1, all_preds, all_labels, per_class_metrics
        else:
            return None, None, None, None, None, None, None

    def calculate_per_class_metrics(self, cm):
        """计算每个类别的指标"""
        n_classes = cm.shape[0]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) \
                if (precision[i] + recall[i]) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self):
        """主训练流程"""
        if self.rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Training PPG + {self.model_type.replace('_', ' ').title()} Model")
            print(f"Using {self.world_size} GPUs with DDP")
            print(f"{'=' * 60}")

        # 准备数据
        data_paths = {
            'ppg': self.config['data']['ppg_file'],
            'ecg': self.config['data']['ecg_file'],
            'index': self.config['data']['index_file']
        }

        if self.rank == 0:
            print(f"\nUsing ECG data from: {data_paths['ecg']}")

        # 创建数据集
        from multimodal_dataset_aligned import MultiModalSleepDataset, PPGOnlyDataset

        train_dataset = MultiModalSleepDataset(
            data_paths, split='train', use_sleepppg_test_set=True
        )
        val_dataset = MultiModalSleepDataset(
            data_paths, split='val', use_sleepppg_test_set=True
        )
        test_dataset = MultiModalSleepDataset(
            data_paths, split='test', use_sleepppg_test_set=True
        )

        # 创建分布式采样器
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        # 创建数据加载器
        # 调整batch_size以适应多GPU
        batch_size_per_gpu = self.config['training']['batch_size']

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=self.config['data']['num_workers'] // self.world_size,
            pin_memory=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=self.config['data']['num_workers'] // self.world_size,
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=self.config['data']['num_workers'] // self.world_size,
            pin_memory=True
        )

        # 创建模型
        model = ImprovedMultiModalSleepNet(
            n_classes=4,
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_fusion_blocks=self.config['model']['n_fusion_blocks']
        ).to(self.device)

        # 使用DDP包装模型
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)

        if self.rank == 0:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 计算类别权重
        class_weights = self.calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # 学习率调度
        total_steps = len(train_loader) * self.config['training']['num_epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['training']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # 训练循环
        best_kappa = 0
        best_epoch = 0
        patience_counter = 0

        train_losses = []
        val_losses = []
        val_kappas = []

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            if self.rank == 0:
                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )

            if self.rank == 0:
                train_losses.append(train_loss)

            # 验证（只在主进程进行）
            if self.rank == 0:
                val_results = self.validate(model.module, val_loader, criterion)
                val_loss, val_acc, val_kappa, val_f1, val_preds, val_labels, per_class = val_results

                val_losses.append(val_loss)
                val_kappas.append(val_kappa)

                print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Val Kappa: {val_kappa:.4f}, Val F1: {val_f1:.4f}")

                # 打印每个类别的性能
                stage_names = ['Wake', 'Light', 'Deep', 'REM']
                print("\nPer-class performance:")
                for i, name in enumerate(stage_names):
                    print(f"  {name}: P={per_class['precision'][i]:.3f}, "
                          f"R={per_class['recall'][i]:.3f}, F1={per_class['f1'][i]:.3f}")

                # 记录到tensorboard
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Acc', train_acc, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Acc', val_acc, epoch)
                self.writer.add_scalar('Val/Kappa', val_kappa, epoch)
                self.writer.add_scalar('Val/F1', val_f1, epoch)

                # 保存最佳模型
                if val_kappa > best_kappa:
                    best_kappa = val_kappa
                    best_epoch = epoch
                    patience_counter = 0

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_kappa': best_kappa,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'config': self.config,
                        'model_type': self.model_type
                    }

                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    torch.save(checkpoint, best_path)
                    print(f"Saved best model with kappa: {best_kappa:.4f}")

                    # 保存混淆矩阵
                    cm = confusion_matrix(val_labels, val_preds)
                    self.plot_confusion_matrix(cm, epoch)
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= self.config['training']['patience']:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

                # 定期保存
                if epoch % self.config['output']['save_frequency'] == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, checkpoint_path)

            # 同步所有进程
            dist.barrier()

            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 测试（只在主进程）
        if self.rank == 0:
            print(f"\nBest validation kappa: {best_kappa:.4f} at epoch {best_epoch}")

            print("\n" + "=" * 60)
            print("Evaluating on test set...")
            print("=" * 60)

            # 加载最佳模型
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'best_model.pth'))
            model.module.load_state_dict(checkpoint['model_state_dict'])

            # 测试
            test_results = self.validate(model.module, test_loader, criterion)
            test_loss, test_acc, test_kappa, test_f1, test_preds, test_labels, test_per_class = test_results

            print(f"\nTest Results:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Kappa: {test_kappa:.4f}")
            print(f"  F1 Score: {test_f1:.4f}")

            # 详细报告
            report = classification_report(
                test_labels, test_preds,
                target_names=['Wake', 'Light', 'Deep', 'REM'],
                output_dict=True
            )

            print("\nClassification Report:")
            print(classification_report(
                test_labels, test_preds,
                target_names=['Wake', 'Light', 'Deep', 'REM']
            ))

            # 保存结果
            results = {
                'model_type': self.model_type,
                'test_accuracy': test_acc,
                'test_kappa': test_kappa,
                'test_f1': test_f1,
                'test_loss': test_loss,
                'best_epoch': best_epoch,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
                'per_class_metrics': {
                    'precision': test_per_class['precision'].tolist(),
                    'recall': test_per_class['recall'].tolist(),
                    'f1': test_per_class['f1'].tolist()
                },
                'config': self.config
            }

            with open(os.path.join(self.results_dir, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

            # 绘制最终混淆矩阵
            self.plot_confusion_matrix(confusion_matrix(test_labels, test_preds), 'final')

            # 绘制训练曲线
            self.plot_training_curves(train_losses, val_losses, val_kappas)

            self.writer.close()

            return results

        return None

    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])

        model_type_title = self.model_type.replace('_', ' ').title()
        plt.title(f'Confusion Matrix - {model_type_title} - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, f'confusion_matrix_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_training_curves(self, train_losses, val_losses, val_kappas):
        """绘制训练曲线"""
        epochs = range(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_kappas, 'g-', label='Val Kappa')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Kappa')
        ax2.set_title('Validation Kappa')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        model_type_title = self.model_type.replace('_', ' ').title()
        fig.suptitle(f'Training Curves - {model_type_title}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300)
        plt.close()


def run_training(rank, world_size, config, run_id):
    """每个进程运行的训练函数"""
    setup(rank, world_size)

    trainer = CrossAttentionTrainerDDP(rank, world_size, config, run_id)
    results = trainer.train()

    cleanup()

    return results


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Cross-Attention Sleep Model with DDP')
    parser.add_argument('--config', type=str, default='config_crossattn_generated.yaml',
                        help='Path to configuration file')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs')
    parser.add_argument('--gpus', type=int, default=3,
                        help='Number of GPUs to use')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置GPU数量
    world_size = min(args.gpus, torch.cuda.device_count())
    print(f"Available GPUs: {torch.cuda.device_count()}, Using: {world_size}")

    # 多次运行
    n_runs = args.runs
    all_results = []

    model_type = config.get('model_type', 'generated_ecg')

    for run in range(1, n_runs + 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {run}/{n_runs}")
        print('=' * 80)

        # 设置随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # 使用多进程启动训练
        if world_size > 1:
            mp.spawn(
                run_training,
                args=(world_size, config, run),
                nprocs=world_size,
                join=True
            )
        else:
            # 单GPU训练
            trainer = CrossAttentionTrainerDDP(0, 1, config, run)
            results = trainer.train()
            if results is not None:
                all_results.append(results)

        # 读取主进程保存的结果
        if world_size > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_dir = f"crossattn_{model_type}_v2_ddp_*_run{run}"
            # 查找结果文件
            import glob
            result_files = glob.glob(os.path.join(config['output']['save_dir'],
                                                  result_dir, 'results', 'test_results.json'))
            if result_files:
                with open(result_files[-1], 'r') as f:
                    results = json.load(f)
                    all_results.append(results)

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

    # 汇总结果
    if all_results:
        print("\n" + "=" * 80)
        print(f"FINAL RESULTS ({len(all_results)} runs)")
        print("=" * 80)

        accuracies = [r['test_accuracy'] for r in all_results]
        kappas = [r['test_kappa'] for r in all_results]
        f1_scores = [r['test_f1'] for r in all_results]

        model_type_title = model_type.replace('_', ' ').title()
        print(f"\nCross-Attention PPG + {model_type_title} Model:")
        print(f"  Accuracy: {np.median(accuracies):.4f} (median), "
              f"{np.mean(accuracies):.4f}±{np.std(accuracies):.4f} (mean±std)")
        print(f"  Kappa: {np.median(kappas):.4f} (median), "
              f"{np.mean(kappas):.4f}±{np.std(kappas):.4f} (mean±std)")
        print(f"  F1 Score: {np.median(f1_scores):.4f} (median), "
              f"{np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f} (mean±std)")
        print(f"  All accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
        print(f"  All kappas: {[f'{k:.4f}' for k in kappas]}")

        # 保存汇总结果
        summary_results = {
            'model_type': model_type,
            'num_runs': len(all_results),
            'num_gpus': world_size,
            'accuracy': {
                'median': float(np.median(accuracies)),
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'all': accuracies
            },
            'kappa': {
                'median': float(np.median(kappas)),
                'mean': float(np.mean(kappas)),
                'std': float(np.std(kappas)),
                'all': kappas
            },
            'f1_score': {
                'median': float(np.median(f1_scores)),
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'all': f1_scores
            },
            'all_runs': all_results
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(config['output']['save_dir'],
                                    f'crossattn_{model_type}_ddp_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2)

        print(f"\nSummary results saved to: {summary_path}")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()