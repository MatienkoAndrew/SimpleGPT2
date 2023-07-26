import torch
import argparse

from config import CONFIG
from tweet_dataset import TweetDataset
from train_and_evaluate import training_model, evaluate_valid, evaluate
from utils import load_data, load_tokenizer, prepare_data, prepare_loader, get_unpretrained_model
from utils import get_pretrained_model, get_attention_matrixes, show_attention, inference 


def get_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--mode', type=str, default='train', help="'train' or 'infer'")
    # parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--text', type=str, default="I have a good day", help='Text for infer model')

    args = parser.parse_args()

    return args


def main():
    dataset = load_data()

    tokenizer = load_tokenizer()

    (train_dataset, valid_dataset, test_dataset) = prepare_data(dataset, tokenizer)

    (train_loader, valid_loader, test_loader) = prepare_loader(train_dataset, valid_dataset, test_dataset)

    # GPT2 для классификации текста (не предобученная)
    model_0 = get_unpretrained_model(tokenizer).to(CONFIG['device'])
    ##-- GPT2 для классификации текста (предобученная)
    model_1 = get_pretrained_model(tokenizer).to(CONFIG['device'])

    args = get_args()
    if args.mode == 'train':
        lr = 1e-5 
        optimizer = torch.optim.AdamW(model_0.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        training_model(model_0, train_loader, valid_loader, optimizer, scheduler)
        evaluate(model_0, test_loader)

        optimizer = torch.optim.AdamW(model_1.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        training_model(model_1, train_loader, valid_loader, optimizer, scheduler)
        evaluate(model_1, test_loader)

    elif args.mode == 'infer':
        model_0.load_state_dict(torch.load('./models/DistilGPT2ForSequenceClassification_notpretrained.pth', map_location=CONFIG['device']))
        model_1.load_state_dict(torch.load('./models/DistilGPT2ForSequenceClassification_pretrained.pth', map_location=CONFIG['device']))

        ##-- inference
        inference(model_0, tokenizer, args.text)
        inference(model_1, tokenizer, args.text)
    else:
        raise f"Unknown mode: {args.mode}. Please use 'train' or 'infer'."

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


# text = dataset["train"]["text"][0] # Выбери текст из датасета
# tokens = tokenizer.tokenize(text)

# attns = get_attention_matrixes(model_0, tokenizer, text)
# show_attention(tokens, attns[-1][0])

# attns = get_attention_matrixes(model_1, tokenizer, text)
# show_attention(tokens, attns[-1][0])