"""
训练改进的PPG+ECG模型 - 使用Cross-Attention (内存优化版)
"""
import torch
import torch.nn as nn
import torch.optim as optim
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

from multimodal_model_crossattn import ImprovedMultiModalSleepNet
from multimodal_dataset_aligned import get_dataloaders


class CrossAttentionTrainer:
    def __init__(self, config, run_id=None):
        self.config = config
        self.run_id = run_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 创建输出目录
        self.setup_directories()

        # 初始化tensorboard
        self.writer = SummaryWriter(self.log_dir)

        # 保存配置
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("Using mixed precision training")

    def setup_directories(self):
        """创建必要的目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"crossattn_ecg_v2_{timestamp}"
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
        """计算类别权重"""
        print("\nCalculating class weights...")

        all_labels = []
        sample_size = min(len(train_dataset), 50)

        for idx in tqdm(range(sample_size), desc="Sampling labels"):
            _, _, labels = train_dataset[idx]
            valid_labels = labels[labels != -1]
            all_labels.extend(valid_labels.numpy().tolist())

        label_counts = Counter(all_labels)
        class_counts = [label_counts.get(i, 1) for i in range(4)]
        total_samples = sum(class_counts)

        print(f"\nLabel distribution:")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            print(f"  {stage_names[i]}: {count} samples ({percentage:.2f}%)")

        # 使用inverse frequency weighting
        class_weights = torch.tensor([total_samples / (4 * count) for count in class_counts],
                                     dtype=torch.float32)

        print(f"\nClass weights: {class_weights}")

        return class_weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion, scheduler, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

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

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 优化器步骤
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            else:
                # 标准训练
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

            # 更新进度条
            if total > 0:
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': correct / total,
                    'lr': optimizer.param_groups[0]['lr']
                })

            # 定期清理内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self, model, dataloader, criterion):
        """验证"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for ppg, ecg, labels in tqdm(dataloader, desc="Validation"):
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

        # 计算指标
        epoch_loss = running_loss / len(all_labels) if all_labels else 0
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if all_labels else 0
        kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0

        # 计算每个类别的指标
        cm = confusion_matrix(all_labels, all_preds)
        per_class_metrics = self.calculate_per_class_metrics(cm)

        return epoch_loss, accuracy, kappa, f1, all_preds, all_labels, per_class_metrics

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
        print(f"\n{'=' * 60}")
        print("Training Improved PPG+ECG Model with Cross-Attention V2")
        print(f"{'=' * 60}")

        # 准备数据
        data_paths = {
            'ppg': self.config['data']['ppg_file'],
            'real_ecg': self.config['data']['ecg_file'],
            'index': self.config['data']['index_file']
        }

        # 创建数据加载器
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
            data_paths,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            model_type='multimodal',
            use_sleepppg_test_set=True
        )

        # 创建模型
        model = ImprovedMultiModalSleepNet(
            n_classes=4,
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_fusion_blocks=self.config['model']['n_fusion_blocks']
        ).to(self.device)

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
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler, epoch
            )
            train_losses.append(train_loss)

            # 验证
            val_loss, val_acc, val_kappa, val_f1, val_preds, val_labels, per_class = self.validate(
                model, val_loader, criterion
            )
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
            self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # 记录每个类别的指标
            for i, name in enumerate(stage_names):
                self.writer.add_scalar(f'Val/Precision_{name}', per_class['precision'][i], epoch)
                self.writer.add_scalar(f'Val/Recall_{name}', per_class['recall'][i], epoch)
                self.writer.add_scalar(f'Val/F1_{name}', per_class['f1'][i], epoch)

            # 保存最佳模型
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_epoch = epoch
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_kappa': best_kappa,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': self.config
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)

            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nBest validation kappa: {best_kappa:.4f} at epoch {best_epoch}")

        # 在测试集上评估
        print("\n" + "=" * 60)
        print("Evaluating on test set...")
        print("=" * 60)

        # 加载最佳模型
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        # 测试
        test_loss, test_acc, test_kappa, test_f1, test_preds, test_labels, test_per_class = self.validate(
            model, test_loader, criterion
        )

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

    def plot_confusion_matrix(self, cm, epoch):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 创建标注
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])

        plt.title(f'Confusion Matrix - Epoch {epoch}')
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

        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Kappa曲线
        ax2.plot(epochs, val_kappas, 'g-', label='Val Kappa')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Kappa')
        ax2.set_title('Validation Kappa')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), dpi=300)
        plt.close()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Cross-Attention Sleep Model')
    parser.add_argument('--config', type=str, default='config_crossattn_v2.yaml',
                        help='Path to configuration file')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs')
    args = parser.parse_args()

    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 默认配置
        config = {
            'data': {
                'ppg_file': "../../data/mesa_ppg_with_labels.h5",
                'ecg_file': "../../data/mesa_real_ecg.h5",
                'index_file': "../../data/mesa_subject_index.h5",
                'num_workers': 4
            },
            'training': {
                'batch_size': 2,
                'num_epochs': 50,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'patience': 15
            },
            'model': {
                'd_model': 256,
                'n_heads': 8,
                'n_fusion_blocks': 3
            },
            'output': {
                'save_dir': "./outputs",
                'save_frequency': 5
            },
            'use_amp': True
        }

    # 多次运行
    n_runs = args.runs
    all_results = []

    for run in range(1, n_runs + 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {run}/{n_runs}")
        print('=' * 80)

        # 设置随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)

        # 训练
        trainer = CrossAttentionTrainer(config, run_id=run)
        results = trainer.train()
        all_results.append(results)

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

    # 汇总结果
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS ({n_runs} runs)")
    print("=" * 80)

    accuracies = [r['test_accuracy'] for r in all_results]
    kappas = [r['test_kappa'] for r in all_results]
    f1_scores = [r['test_f1'] for r in all_results]

    print(f"\nCross-Attention PPG+ECG Model V2:")
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
        'num_runs': n_runs,
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
                                f'crossattn_v2_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nSummary results saved to: {summary_path}")


if __name__ == "__main__":
    main()