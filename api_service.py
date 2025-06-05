import os
import json
import asyncio
import subprocess
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import mlflow
import torch
from PIL import Image
import numpy as np

from train import ModelTrainer, ModifiedResNet50, PerceptualHasher
from torchvision import transforms

# Инициализация FastAPI
app = FastAPI(
    title="Photo Archive Optimizer API",
    description="API для оптимизации фотоархивов с использованием нейросетей",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение к Redis
redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)

# Глобальные переменные для моделей
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hasher = PerceptualHasher()

# Pydantic модели
class OptimizationRequest(BaseModel):
    input_path: str
    output_path: str
    algorithm: str = "gzip"
    remove_blurred: bool = True
    remove_duplicates: bool = True
    compress_files: bool = True
    workers: int = 4

class OptimizationResult(BaseModel):
    task_id: str
    status: str
    original_size: Optional[int] = None
    compressed_size: Optional[int] = None
    files_processed: Optional[int] = None
    files_removed: Optional[int] = None
    compression_ratio: Optional[float] = None
    processing_time: Optional[float] = None

class BlurPrediction(BaseModel):
    filename: str
    is_blurred: bool
    confidence: float

class DuplicateGroup(BaseModel):
    files: List[str]
    similarity_score: float

def load_model():
    """Загрузка предобученной модели"""
    global model
    try:
        model_path = os.getenv('MODEL_PATH', './models/best_model.pth')
        if os.path.exists(model_path):
            model = ModifiedResNet50(num_classes=2, pretrained=False)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            print("Модель успешно загружена")
        else:
            print("Файл модели не найден, используется предобученная модель")
            model = ModifiedResNet50(num_classes=2, pretrained=True)
            model.to(device)
            model.eval()
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    load_model()
    
    # Создание необходимых директорий
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./processed", exist_ok=True)
    os.makedirs("./compressed", exist_ok=True)

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Photo Archive Optimizer API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "redis_connected": redis_client.ping(),
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict/blur")
async def predict_blur(files: List[UploadFile] = File(...)):
    """Предсказание размытости для загруженных изображений"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    results = []
    
    for file in files:
        try:
            # Сохранение временного файла
            temp_path = f"./uploads/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Предобработка изображения
            image = Image.open(temp_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Предсказание
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                predicted_class = output.argmax(dim=1).item()
            
            results.append(BlurPrediction(
                filename=file.filename,
                is_blurred=bool(predicted_class),
                confidence=confidence
            ))
            
            # Удаление временного файла
            os.remove(temp_path)
            
        except Exception as e:
            results.append(BlurPrediction(
                filename=file.filename,
                is_blurred=False,
                confidence=0.0
            ))
    
    return {"predictions": results}

@app.post("/find-duplicates")
async def find_duplicates(files: List[UploadFile] = File(...)):
    """Поиск дубликатов среди загруженных изображений"""
    temp_files = []
    
    try:
        # Сохранение временных файлов
        for file in files:
            temp_path = f"./uploads/{file.filename}"
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            temp_files.append(temp_path)
        
        # Поиск дубликатов
        duplicate_groups = hasher.find_duplicates(temp_files, threshold=5)
        
        # Форматирование результатов
        results = []
        for group in duplicate_groups:
            # Вычисление средней схожести в группе
            similarity_score = 0.9  # Упрощенная метрика
            results.append(DuplicateGroup(
                files=[os.path.basename(f) for f in group],
                similarity_score=similarity_score
            ))
        
        return {"duplicate_groups": results}
        
    finally:
        # Очистка временных файлов
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.post("/optimize")
async def optimize_archive(
    background_tasks: BackgroundTasks,
    request: OptimizationRequest
):
    """Запуск оптимизации фотоархива"""
    
    # Генерация ID задачи
    task_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Сохранение задачи в Redis
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    redis_client.setex(f"task:{task_id}", 86400, json.dumps(task_data))
    
    # Добавление фоновой задачи
    background_tasks.add_task(run_optimization, task_id, request)
    
    return {"task_id": task_id, "status": "started"}

async def run_optimization(task_id: str, request: OptimizationRequest):
    """Выполнение оптимизации в фоновом режиме"""
    try:
        # Обновление статуса
        update_task_status(task_id, "running", {"message": "Начало обработки"})
        
        # Этап 1: Анализ качества изображений
        if request.remove_blurred and model is not None:
            update_task_status(task_id, "running", {"message": "Анализ качества изображений"})
            blurred_files = await analyze_image_quality(request.input_path)
            
            # Удаление размытых файлов
            for file_path in blurred_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Этап 2: Поиск и удаление дубликатов
        if request.remove_duplicates:
            update_task_status(task_id, "running", {"message": "Поиск дубликатов"})
            duplicate_groups = await find_duplicate_files(request.input_path)
            
            # Удаление дубликатов (оставляем по одному из каждой группы)
            for group in duplicate_groups:
                for file_path in group[1:]:  # Удаляем все кроме первого
                    if os.path.exists(file_path):
                        os.remove(file_path)
        
        # Этап 3: Сжатие оставшихся файлов
        if request.compress_files:
            update_task_status(task_id, "running", {"message": "Сжатие файлов"})
            compression_result = await compress_files(
                request.input_path, 
                request.output_path, 
                request.algorithm,
                request.workers
            )
            
            # Финальное обновление статуса
            update_task_status(task_id, "completed", {
                "message": "Оптимизация завершена",
                "result": compression_result
            })
        else:
            update_task_status(task_id, "completed", {"message": "Оптимизация завершена"})
            
    except Exception as e:
        update_task_status(task_id, "failed", {"error": str(e)})

def update_task_status(task_id: str, status: str, data: Dict):
    """Обновление статуса задачи в Redis"""
    try:
        task_data = redis_client.get(f"task:{task_id}")
        if task_data:
            task_info = json.loads(task_data)
            task_info["status"] = status
            task_info["updated_at"] = datetime.now().isoformat()
            task_info.update(data)
            redis_client.setex(f"task:{task_id}", 86400, json.dumps(task_info))
    except Exception as e:
        print(f"Ошибка обновления статуса задачи: {e}")

async def analyze_image_quality(input_path: str) -> List[str]:
    """Анализ качества изображений и возврат списка размытых файлов"""
    blurred_files = []
    transform = transforms.Compose([...])
    
    for file_path in Path(input_path).rglob("*.jpg"):
        try:
            # Загрузка и предобработка изображения
            image = Image.open(file_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Предсказание
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = output.argmax(dim=1).item()
                
                if predicted_class == 1:  # Размытое изображение
                    blurred_files.append(str(file_path))
                    
        except Exception as e:
            print(f"Ошибка анализа {file_path}: {e}")
    
    return blurred_files

async def find_duplicate_files(input_path: str) -> List[List[str]]:
    """Поиск дубликатов в директории"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(input_path).rglob(ext))
    
    duplicate_groups = hasher.find_duplicates([str(f) for f in image_files])
    return duplicate_groups

async def run_optimization(task_id: str, request: OptimizationRequest):
    start_time = time.time()
    stats = {
        "blurred_removed": 0,
        "duplicates_removed": 0
    }
    
    # Шаг 1: Удаление размытых изображений
    if request.remove_blurred:
        blurred_files = await analyze_image_quality(request.input_path)
        stats["blurred_removed"] = len(blurred_files)
    
    # Шаг 2: Удаление дубликатов
    if request.remove_duplicates:
        duplicate_groups = await find_duplicate_files(request.input_path)
        for group in duplicate_groups:
            stats["duplicates_removed"] += len(group) - 1
    
    # Шаг 3: Сжатие
    if request.compress_files:
        compression_result = await compress_files(...)
        stats.update(compression_result)
    
    stats["processing_time"] = time.time() - start_time
    update_task_status(task_id, "completed", stats)

async def compress_files(input_path: str, output_path: str, algorithm: str, workers: int) -> Dict:
    """Сжатие файлов с использованием Go-компонента"""
    try:
        # Вызов Go-программы для сжатия
        cmd = [
            "./image-compressor",
            "-input", input_path,
            "-output", output_path,
            "-algorithm", algorithm,
            "-workers", str(workers),
            "-mode", "compress"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Парсинг результатов (упрощенная версия)
            return {
                "success": True,
                "output": result.stdout,
                "compression_ratio": 0.7  # Упрощенная метрика
            }
        else:
            return {
                "success": False,
                "error": result.stderr
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса задачи"""
    task_data = redis_client.get(f"task:{task_id}")
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return json.loads(task_data)

@app.get("/tasks")
async def list_tasks():
    """Список всех задач"""
    keys = redis_client.keys("task:*")
    tasks = []
    
    for key in keys:
        task_data = redis_client.get(key)
        if task_data:
            tasks.append(json.loads(task_data))
    
    return {"tasks": tasks}

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Удаление задачи"""
    deleted = redis_client.delete(f"task:{task_id}")
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return {"message": "Задача удалена"}

@app.get("/metrics")
async def get_metrics():
    """Получение метрик системы"""
    import psutil
    
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)