import torch
from torch.nn.functional import softmax
from transformers import GPT2ForSequenceClassification, GPT2TokenizerFast, GPT2Config
from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm.auto import tqdm

from tweet_dataset import TweetDataset
from config import CONFIG

def load_data():
    """
    Загрузка набора данных "emotion" из библиотеки "datasets".
    Возвращает загруженный набор данных.
    """
    dataset = load_dataset("emotion")
    return dataset


def load_tokenizer(model_name: str='distilgpt2'):
    """
    Загрузка токенизатора для заданной модели.
    Возвращает загруженный токенизатор.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # У gpt2 нет pad токенов. Вместо них воспользуемся токенами конца текста.
    return tokenizer


def prepare_data(dataset, tokenizer) -> tuple:
    """
    Подготовка данных для обучения, валидации и тестирования.
    Возвращает кортеж из трех объектов типа TweetDataset.
    """
    train_dataset = TweetDataset(part='train', dataset=dataset, tokenizer=tokenizer)
    valid_dataset = TweetDataset('validation', dataset=dataset, tokenizer=tokenizer) # validation
    test_dataset = TweetDataset('test', dataset=dataset, tokenizer=tokenizer)
    return (train_dataset, valid_dataset, test_dataset)


def prepare_loader(train_dataset, valid_dataset, test_dataset) -> tuple:
    """
    Подготовка загрузчиков данных для обучения, валидации и тестирования.
    Возвращает кортеж из трех объектов типа DataLoader.
    """
    batch_size = CONFIG['batch_size']
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    return (train_loader, valid_loader, test_loader)


def get_unpretrained_model(tokenizer):
    """
    Создание необученной модели с конфигурацией GPT2.
    Возвращает модель.
    """
    config = GPT2Config.from_pretrained(
        "distilgpt2",
        output_attentions=True,
        pad_token_id=tokenizer.eos_token_id,
        num_labels=6
    )
    model = GPT2ForSequenceClassification(config=config)
    return model

def get_pretrained_model(tokenizer):
    """
    Загрузка предобученной модели с конфигурацией GPT2.
    Возвращает модель.
    """
    model = GPT2ForSequenceClassification.from_pretrained(
        "distilgpt2", 
        output_attentions=True,
        pad_token_id=tokenizer.eos_token_id,
        num_labels=6
    )
    return model


def inference(model, tokenizer, text: str):
    """
    Инференс модели на заданном тексте.
    Печатает предсказанную эмоцию.
    """
    map_classes = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    encoded_input = tokenizer.encode_plus(
                text, 
                max_length=CONFIG['max_length'], # максимальная длина текста
                padding="max_length", # надо ли добавлять паддинг в конце?
                return_tensors="pt", # возвращает pytorch тензоры
            ).to(CONFIG['device'])
    outputs = model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])

    logits = outputs.logits
    probabilities = softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()

    print(f'Predicted emotion: {map_classes[predicted_class]}')


def get_attention_matrixes(model, tokenizer, text):
    """
    Получение матриц внимания модели для заданного текста.
    Возвращает массив numpy с матрицами внимания.
    """
    inp = list(filter(lambda x: x != tokenizer.sep_token_id, tokenizer.encode(text)))
    inp = torch.tensor(inp, dtype=torch.long, device=CONFIG['device']).unsqueeze(0)
    attn_tensors = model(inp)[-1]
    seq = [tokenizer.decode(x) for x in inp[0].tolist()]
    attn = []
    for i in range(len(attn_tensors)):
        attn_layer = []
        for j in range(attn_tensors[i].size(1)):
            attn_layer.append(attn_tensors[i][0, j].cpu().detach().numpy())
        attn.append(np.array(attn_layer))
    
    return np.array(attn)


def show_attention(seq, attentions):
    """
    Отображение визуализации матриц внимания.
    """
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(['']+seq, rotation=90, fontsize=16)
    ax.set_yticklabels(['']+seq, fontsize=16)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
