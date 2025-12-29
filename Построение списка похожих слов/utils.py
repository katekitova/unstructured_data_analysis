import os
import urllib
import requests
import zipfile
from urllib.request import urlretrieve
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

BASE_YADISK_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
VARIANT_MAPPING = {
    1: "https://disk.yandex.ru/d/4NbkdrzBs6np_g",
    2: "https://disk.yandex.ru/d/15hy2ttrbAk8fg",
    3: "https://disk.yandex.ru/d/2yZ-80F7R-8nww",
    4: "https://disk.yandex.ru/d/o7ry8XJUGzFOkw",
    5: "https://disk.yandex.ru/d/H6NxkdxepWZYPg",
    6: "https://disk.yandex.ru/d/IHD9Id1QfylhKw",
    7: "https://disk.yandex.ru/d/VpSXG4u3VMI9MQ",
    8: "https://disk.yandex.ru/d/_SF_EQrRew3mOw",
}


def download_model(data_dir: str='data') -> None:
    """
    Скачивает и распаковывает модель в указанную директорию.
    
    Args:
        data_dir (str): Путь к основной директории данных
    """
    # Создаем Path объекты для директорий
    data_path = Path(data_dir)
    model_path = data_path / 'model'
    
    model_url = "http://vectors.nlpl.eu/repository/20/213.zip"
    zip_filename = "213.zip"
    zip_path = model_path / zip_filename
    
    try:
        # Создаем директории
        data_path.mkdir(parents=True, exist_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)

        # Проверка, что уже скачали архив
        if not zip_path.exists():
            print(f"Скачиваем архив из {model_url}...")
            urlretrieve(model_url, zip_path)
            print("Архив успешно скачан")
        
        # Распаковываем архив
        print("Распаковываем архив...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_path)
        print("Архив успешно распакован")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def download_words(variant: int) -> pd.DataFrame:
    assert variant in VARIANT_MAPPING.keys(), 'Incorrect vatiant number'
    url = VARIANT_MAPPING[variant]

    final_url = BASE_YADISK_URL + urllib.parse.urlencode(dict(public_key=url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    df = pd.read_csv(download_url)
    
    return df


def download_paraphrases() -> pd.DataFrame:

    url = "https://disk.yandex.ru/d/0uhKEwnobwWivA"

    final_url = BASE_YADISK_URL + urllib.parse.urlencode(dict(public_key=url))
    response = requests.get(final_url)
    download_url = response.json()['href']

    df = pd.read_csv(download_url)

    return df


def plot_clusters_with_labels(embeddings: np.ndarray,
                            labels: np.ndarray,
                            words: List[str],
                            figsize: Tuple[int, int] = (20, 20),
                            sample_size: Optional[int] = None,
                            random_state: int = 42) -> plt.Figure:
    """
    Визуализирует кластеры с подписями слов используя t-SNE проекцию.
    
    Args:
        embeddings: numpy array размера (n_samples, n_features)
        labels: numpy array размера (n_samples,) с метками кластеров
        words: список слов длины n_samples
        figsize: размер графика
        sample_size: количество случайных точек для отображения (None = все точки)
        random_state: seed для воспроизводимости
    
    Returns:
        fig: объект figure с визуализацией
    """
    # Уменьшаем размерность до 2D используя t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(min([labels.count(i) for i in set(labels)]), embeddings.shape[0]))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Если указан размер выборки, делаем случайную выборку
    if sample_size is not None and sample_size < len(words):
        np.random.seed(random_state)
        indices = np.random.choice(len(words), sample_size, replace=False)
        embeddings_2d = embeddings_2d[indices]
        labels = labels[indices]
        words = [words[i] for i in indices]
    
    # Создаем график
    plt.figure(figsize=figsize)
    
    # Определяем цвета для кластеров (используем встроенную цветовую карту)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Создаем scatter plot
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], 
                   embeddings_2d[mask, 1],
                   c=[color], 
                   label=f'Cluster {label}',
                   alpha=0.6)
    
    # Добавляем подписи к точкам
    for i, word in enumerate(words):
        plt.annotate(word,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8)
    
    # Настраиваем график
    plt.title("Clusters Visualization (t-SNE projection)", fontsize=16, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()


def find_best_accuracy(
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        n_thresholds: int = 100
    ) -> Tuple[float, float]:
    """
    Находит оптимальный порог, максимизирующий accuracy
    
    Args:
        y_true: истинные метки классов (0 или 1)
        y_pred_proba: предсказанные вероятности принадлежности к классу 1
        n_thresholds: количество порогов для проверки
        
    Returns:
        Tuple[float, float]: (оптимальный_порог, лучший_accuracy)
    """
    # Генерируем различные пороги для проверки
    thresholds = np.linspace(0, 1, n_thresholds)
    
    # Сохраняем лучший результат
    best_accuracy = 0
    
    # Перебираем все пороги
    for threshold in thresholds:
        # Получаем предсказания для текущего порога
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Вычисляем accuracy
        current_accuracy = accuracy_score(y_true, y_pred)
        
        # Обновляем лучший результат если нужно
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            
    return best_accuracy
