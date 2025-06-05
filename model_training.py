import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.cluster import DBSCAN
import cv2
import imagehash

import mlflow
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger
import dvc.api

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlurDetectionDataset(Dataset):
    """
    Датасет для обнаружения размытых изображений
    Поддерживает CUHK Blur Detection Dataset и REDS
    """
    
    def __init__(self, data_dir: str, annotations_file: str = None, 
                 transform=None, mode='blur_detection'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        self.samples = []
        
        if annotations_file and os.path.exists(annotations_file):
            self._load_annotated_data(annotations_file)
        else:
            self._load_directory_structure()
    
    def _load_annotated_data(self, annotations_file: str):
        """Загрузка данных с аннотациями"""
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        for item in annotations:
            img_path = self.data_dir / item['image']
            if img_path.exists():
                label = 1 if item['blur'] else 0  # 1 - размытое, 0 - четкое
                self.samples.append((str(img_path), label))
    
    def _load_directory_structure(self):
        """Загрузка данных по структуре директорий (sharp/blur)"""
        sharp_dir = self.data_dir / 'sharp'
        blur_dir = self.data_dir / 'blur'
        
        if sharp_dir.exists():
            for img_path in sharp_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), 0))  # 0 - четкое
        
        if blur_dir.exists():
            for img_path in blur_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), 1))  # 1 - размытое
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.warning(f"Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем черное изображение в случае ошибки
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label


class AdvancedAugmentation:
    """Продвинутые аугментации для повышения устойчивости модели"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            # Случайное изменение яркости и контраста
            brightness_factor = np.random.uniform(0.8, 1.2)
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
            
            # Случайное добавление шума
            if np.random.random() < 0.3:
                noise = torch.randn_like(TF.to_tensor(image)) * 0.01
                image = TF.to_pil_image(TF.to_tensor(image) + noise)
        
        return image


class ModifiedResNet50(nn.Module):
    """
    Модифицированная версия ResNet50 для классификации качества изображений
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        super(ModifiedResNet50, self).__init__()
        
        # Загружаем предобученную ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Заменяем последний слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Добавляем слой для извлечения признаков
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
    
    def forward(self, x, return_features=False):
        if return_features:
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)
            return features
        
        return self.backbone(x)


class PerceptualHasher:
    """Класс для вычисления перцептивных хешей и поиска дубликатов"""
    
    def __init__(self, hash_size=8):
        self.hash_size = hash_size
    
    def compute_hash(self, image_path: str) -> str:
        """Вычисление pHash для изображения"""
        try:
            image = Image.open(image_path)
            return str(imagehash.phash(image, hash_size=self.hash_size))
        except Exception as e:
            logger.warning(f"Ошибка вычисления хеша для {image_path}: {e}")
            return ""
    
    def find_duplicates(self, image_paths: List[str], threshold: int = 5) -> List[List[str]]:
        """Поиск дубликатов с использованием DBSCAN кластеризации"""
        
        # Вычисляем хеши для всех изображений
        hashes = []
        valid_paths = []
        
        for path in image_paths:
            hash_str = self.compute_hash(path)
            if hash_str:
                hashes.append([int(c, 16) for c in hash_str])
                valid_paths.append(path)
        
        if len(hashes) < 2:
            return []
        
        # Используем DBSCAN для кластеризации похожих хешей
        X = np.array(hashes)
        clustering = DBSCAN(eps=threshold, min_samples=2, metric='hamming')
        labels = clustering.fit_predict(X)
        
        # Группируем дубликаты
        duplicates = {}
        for i, label in enumerate(labels):
            if label != -1:  # -1 означает шум (не дубликат)
                if label not in duplicates:
                    duplicates[label] = []
                duplicates[label].append(valid_paths[i])
        
        return list(duplicates.values())


class SSIMCalculator:
    """Калькулятор SSIM для оценки качества изображений"""
    
    @staticmethod
    def calculate_ssim(img1_path: str, img2_path: str) -> float:
        """Вычисление SSIM между двумя изображениями"""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Приводим к одному размеру
            height, width = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))
            
            # Вычисляем SSIM
            from skimage.metrics import structural_similarity
            ssim_value = structural_similarity(img1, img2)
            
            return ssim_value
        except Exception as e:
            logger.warning(f"Ошибка вычисления SSIM: {e}")
            return 0.0


class ModelTrainer:
    """Основной класс для обучения модели"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Настройка MLflow
        mlflow.set_experiment(config.get('experiment_name', 'photo_archive_optimization'))
        
        logger.info(f"Используется устройство: {self.device}")
    
    def setup_data_transforms(self):
        """Настройка трансформаций данных"""
        
        # Трансформации для обучения (с аугментацией)
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            AdvancedAugmentation(p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформации для валидации/тестирования
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def setup_datasets(self):
        """Настройка датасетов"""
        train_transforms, val_transforms = self.setup_data_transforms()
        
        # Создаем полный датасет
        full_dataset = BlurDetectionDataset(
            data_dir=self.config['data_dir'],
            annotations_file=self.config.get('annotations_file'),
            transform=train_transforms
        )
        
        # Разделяем на train/val/test
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Применяем разные трансформации для валидации и теста
        val_dataset.dataset.transform = val_transforms
        test_dataset.dataset.transform = val_transforms
        
        # Создаем DataLoader'ы
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        logger.info(f"Размеры датасетов - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def setup_model(self):
        """Настройка модели"""
        self.model = ModifiedResNet50(
            num_classes=self.config.get('num_classes', 2),
            pretrained=True,
            dropout_rate=self.config.get('dropout_rate', 0.5)
        ).to(self.device)
        
        logger.info(f"Модель создана с {sum(p.numel() for p in self.model.parameters())} параметрами")
    
    def train_epoch(self, epoch: int, optimizer, criterion, scheduler=None):
        """Обучение на одной эпохе"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Валидация модели"""
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = correct_predictions / total_samples
        
        # Вычисляем дополнительные метрики
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        return val_loss, val_acc, precision, recall, f1
    
    def train(self):
        """Основной цикл обучения"""
        with mlflow.start_run():
            # Логируем параметры
            mlflow.log_params(self.config)
            
            # Настройка оптимизатора и планировщика
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
            criterion = nn.CrossEntropyLoss()
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            best_val_acc = 0.0
            patience_counter = 0
            max_patience = self.config.get('early_stopping_patience', 10)
            
            for epoch in range(self.config['num_epochs']):
                # Обучение
                train_loss, train_acc = self.train_epoch(epoch, optimizer, criterion)
                
                # Валидация
                val_loss, val_acc, precision, recall, f1 = self.validate(criterion)
                
                # Обновление планировщика
                scheduler.step(val_loss)
                
                # Логирование метрик
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_precision': precision,
                    'val_recall': recall,
                    'val_f1': f1,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Сохранение лучшей модели
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_model(f'best_model_epoch_{epoch}.pth')
                    mlflow.pytorch.log_model(self.model, "best_model")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= max_patience:
                    logger.info(f'Early stopping на эпохе {epoch}')
                    break
            
            # Финальное тестирование
            test_metrics = self.test()
            mlflow.log_metrics(test_metrics)
            
            logger.info(f"Обучение завершено. Лучшая точность на валидации: {best_val_acc:.4f}")
    
    def test(self) -> Dict[str, float]:
        """Тестирование модели"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Вычисляем метрики
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Матрица путаницы
        cm = confusion_matrix(all_targets, all_predictions)
        
        test_metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }
        
        logger.info(f"Результаты тестирования: {test_metrics}")
        logger.info(f"Матрица путаницы:\n{cm}")
        
        return test_metrics
    
    def save_model(self, filename: str):
        """Сохранение модели"""
        model_dir = Path(self.config.get('model_save_dir', './models'))
        model_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, model_dir / filename)
        
        logger.info(f"Модель сохранена: {model_dir / filename}")
    
    def load_model(self, model_path: str):
        """Загрузка модели"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Модель загружена: {model_path}")


def setup_dvc_pipeline():
    """Настройка DVC пайплайна для управления данными"""
    
    dvc_yaml = """
stages:
  prepare_data:
    cmd: python prepare_data.py
    deps:
    - data/raw/
    outs:
    - data/processed/
    
  train_model:
    cmd: python train_model.py --config config.json
    deps:
    - data/processed/
    - src/model_training.py
    params:
    - train.learning_rate
    - train.batch_size
    - train.num_epochs
    outs:
    - models/
    metrics:
    - metrics.json
"""
    
    with open('dvc.yaml', 'w') as f:
        f.write(dvc_yaml)
    
    logger.info("DVC пайплайн настроен")


def main():
    parser = argparse.ArgumentParser(description='Обучение модели классификации качества изображений')
    parser.add_argument('--config', type=str, required=True, help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'find_duplicates'])
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.mode == 'train':
        # Обучение модели
        trainer = ModelTrainer(config)
        trainer.setup_datasets()
        trainer.setup_model()
        trainer.train()
        
    elif args.mode == 'test':
        # Тестирование модели
        trainer = ModelTrainer(config)
        trainer.setup_datasets()
        trainer.setup_model()
        trainer.load_model(config['model_path'])
        trainer.test()
        
    elif args.mode == 'find_duplicates':
        # Поиск дубликатов
        hasher = PerceptualHasher()
        image_paths = list(Path(config['data_dir']).rglob('*.jpg'))
        duplicates = hasher.find_duplicates([str(p) for p in image_paths])
        
        logger.info(f"Найдено {len(duplicates)} групп дубликатов:")
        for i, group in enumerate(duplicates):
            logger.info(f"Группа {i+1}: {len(group)} изображений")
            for img in group:
                logger.info(f"  - {img}")


if __name__ == "__main__":
    main()
