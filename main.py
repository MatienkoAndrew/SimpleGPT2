import torch

from config import CONFIG
from tweet_dataset import TweetDataset
from model_training import training_model, evaluate_valid, evaluate
from utils import load_data, load_tokenizer, prepare_data, prepare_loader, get_untrained_model
from utils import get_trained_model, get_attention_matrixes, show_attention, inference 


def main():
    dataset = load_data()

    tokenizer = load_tokenizer()

    (train_dataset, valid_dataset, test_dataset) = prepare_data(dataset, tokenizer)

    (train_loader, valid_loader, test_loader) = prepare_loader(train_dataset, valid_dataset, test_dataset)

    # GPT2 для классификации текста (необученная)
    model_0 = get_untrained_model(tokenizer).to(CONFIG['device'])
    model_0.load_state_dict(torch.load('./models/DistilGPT2ForSequenceClassification_notpretrained.pth', map_location=CONFIG['device']))
    # print(model_0)

    # lr = 1e-5 
    # optimizer = torch.optim.AdamW(model_0.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # training_model(model_0, train_loader, valid_loader, optimizer, scheduler, num_epochs=2)
    evaluate(model_0, test_loader)


    text = dataset["train"]["text"][0] # Выбери текст из датасета
    tokens = tokenizer.tokenize(text)

    attns = get_attention_matrixes(model_0, tokenizer, text)
    show_attention(tokens, attns[-1][0])


    ##-- model_1
    model_1 = get_trained_model(tokenizer).to(CONFIG['device'])
    model_1.load_state_dict(torch.load('./models/DistilGPT2ForSequenceClassification_pretrained.pth', map_location=CONFIG['device']))
    # print(model_1)

    # lr = 1e-5 # Предполагаемый learning rate. Он может быть больше или меньше :)
    # optimizer = torch.optim.AdamW(model_1.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Можно добавить шедулер для обучения моделей. Это на твое усмотрение

    # training_model(model_1, train_loader, valid_loader, optimizer, scheduler, num_epochs=2)
    evaluate(model_1, test_loader)

    attns = get_attention_matrixes(model_1, tokenizer, text)
    show_attention(tokens, attns[-1][0])

    ##-- inference
    inference(model_0, tokenizer, 'I have a good day')
    inference(model_1, tokenizer, 'I have a good day')

    return 0

if __name__ == '__main__':
    main()

##-- examples
# for (emotion, emotion_label) in zip(np.unique(emotion_dataset["train"]["label"]), ["sadness", "joy", "love", "anger", "fear", "surprise"]):
#     print(f"Emotion: {emotion_label}")
#     print("Examples:")
#     indices = np.where(np.array(emotion_dataset["train"]["label"]) == emotion)[0]
#     for text in np.array(emotion_dataset["train"]["text"])[indices[:5]]:
#         print(f"- {text}")
#     print()

