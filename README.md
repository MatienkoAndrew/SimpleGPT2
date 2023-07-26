# SimpleGPT2

Это проект по классификации эмоций в тексте твитов с использованием модели DistilGPT-2 от Hugging Face. 
Модель тренируется на датасете 'emotion' из библиотеки datasets от Hugging Face.

## Структура проекта

```bash
├── [homework_part1]GPT (1).ipynb 
├── config.py # файл, содержащий конфигурационные параметры, используемые в различных файлах проекта.
├── main.py # основной файл, который использует все вышеупомянутые файлы для тренировки модели и оценки ее производительности.
├── train_and_evaluate.py # файл, содержащий функции для обучения модели и оценки ее производительности.
├── models
│   ├── DistilGPT2ForSequenceClassification_notpretrained.onnx
│   ├── DistilGPT2ForSequenceClassification_notpretrained.pth
│   ├── DistilGPT2ForSequenceClassification_pretrained.onnx
│   └── DistilGPT2ForSequenceClassification_pretrained.pth
├── tweet_dataset.py # файл, содержащий определение класса TweetDataset, который наследуется от класса torch.utils.data.Dataset и предоставляет средства для загрузки и предварительной обработки датасета.
└── utils.py # полезные функции для подготовки данных
└── requirements.txt       # файл со списком зависимостей
```

##  Использование

1. Убедитесь, что у вас установлены все необходимые библиотеки из файла requirements.txt. Вы можете установить их, выполнив следующую команду:
```
pip install -r requirements.txt
```
2. Запустите файл main.py, чтобы начать обучение и генерацию текста:
2.1 Обучение модели:
   ```
   python main.py --mode train
   ```
2.2 Инференс модели (выдает предсказанную эмоцию по входному тексту):
  ```
  python main.py --mode infer --text 'What a good day'
  ```

