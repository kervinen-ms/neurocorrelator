# Neurocorrelator

## Постановка задачи

Нейрокоррелятор решает задачу кросс-ракурсного сопоставления шаблонов для геолокализации. Применяется классическая идея
сопоставления на основе кросс-корреляции патчей, но вместо сырых изображений используются их глубокоуровневые нейросетевые представления.

## Формат входных и выходных данных

Входные данные:

-   Изображение эталонного объекта.

-   Изображение сцены.

Выходные данные:

-   Heatmap предсказаний

-   Bounding boxes предсказанного эталона (или нескольких)

### Об архитектуре

-   Базовая модель -- сверточная нейронная сеть VGGNet19, в качестве признаков изображений, используются выходы её со второго и шестнадцатого слоев. Такой выбор обусловлен желанием использовать одновременно и низкоуровневое представление, близкое к сырому изображению, и высокоуровневое, то есть семантические признаки
-   Для ранжирование используется кросс-корреляция патчей и SoftRank
-   Применяется дообучение на датасете Univeristy-1652, состоящем из наборов изображений одних и тех же зданий кампусов с разных ракурсов
-   Для дообучения используется TripletMarginLoss

[cplusx/QATM: Code for Quality-Aware Template Matching for Deep Learning](https://github.com/cplusx/QATM)
_Имплементация алгоритма из оригинальной статьи_

## Setup

### Зависимости

Описаны в соответствующем файле `pyproject.toml`

### Клонирование репозитория

```
git clone https://github.com/yourusername/neurocorrelator.git
cd neurocorrelator
```

### Установка зависимостей

```
poetry install
```

Вот тут непонятно, как загружать данные, если для гугл диска нужен приватный ключ. Инструкций на этот счет в тексте задания нет((( Связаться с автором проекта можно через телеграм, который я указал в форме с отправкой задания в поле _"Ваш отзыв о задании"_. Альтернативный способ -- скачать данные по прямой ссылке, которую я тоже указал в форме.

### Загрузка данных через DVC

```
dvc pull
```

Данные быть в папке `data/`

### Настройка окружения

```
poetry config virtualenvs.create true
virtualenvs.in-project = true
poetry env activate
# скопировать и выполнить выведенную команду
```

### Установить pre-commit хуки

```
pre-commit install
```

## Train

Команда для запуска python может быть другой, т.к. при работе использовались jupyter-notebooks в среде VSCode.

```
python train.py
```

Параметры обучения можно посмотреть в соответствующем конфиге. Во время проверки задания, лосс, скорее всего не сойдется, потому
что таков выбранный метод, он требует много эпох. Но прогресс можно отследить по реколлу

## Inference

Целевое изображение -- 'samples/image.png'
Директория с шаблона и -- 'templates/'
(Эти опции можно поменять, см. -- определение `qatm.py`)

```
python qatm.py
```

После этого результат работы будет в директории `results/`
В нем будет три файла

-   result.png -- целевое изображение с bounding boxes и qatm score
-   heatmap.png -- тепловая карта qatm score
-   boxes.csv -- датасет с bounding boxes

Пушить на гитхаб картинки или любые другие файлы, плохо, конечно, но это необходимо для демонстрации результата

## Структура проекта

neurocorrelator/
├── configs/ # Конфигурации обучения и инференса
├── data/ # Датасеты (управляются через DVC)
├── scripts/ # Вспомогательные скрипты
├── src/ # Исходный код
│ ├── neurocorrelator # Архитектуры сетей
│ │ ├── ...
├── tests/ # Юнит-тесты (тут пусто)
├── .dvc/ # Контроль версий данных
├── poetry.lock # Фиксация версий зависимостей
├── pyproject.toml # Конфигурация проекта
└── train.py # Основной скрипт обучения
└── qatm.py # Основной скрипт инференса
