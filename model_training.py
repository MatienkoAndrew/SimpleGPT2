import torch
from tqdm.auto import tqdm

from config import CONFIG


def training_model(model, train_loader, valid_loader, optimizer, scheduler, num_epochs: int=2):
    # Train loop
    for e in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Training epoch {e}'):
            optimizer.zero_grad()
            loss = model(
                batch['input_ids'].squeeze().to(CONFIG['device']), 
                attention_mask=batch['mask'].to(CONFIG['device']),
                labels=batch['target'].to(CONFIG['device'])
                )['loss']
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            torch.cuda.empty_cache()

        
        valid_loss, valid_acc = evaluate_valid(model, valid_loader)

        scheduler.step()
        print(f"Train Loss: {train_loss / len(train_loader)}")
    
              


def evaluate_valid(model, valid_loader):
    valid_loss = 0
    valid_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            output = model(
                batch['input_ids'].squeeze().to(CONFIG['device']),
                attention_mask = batch['mask'].to(CONFIG['device']),
                labels = batch['target'].to(CONFIG['device'])
            )
            valid_loss += output['loss'].item()
            valid_acc += (output['logits'].argmax(1) == batch['target'].to(CONFIG['device'])).sum().item()

            torch.cuda.empty_cache()

        print(f"Valid Loss: {valid_loss / len(valid_loader)},"
              f"Valid Acc: {valid_acc / len(valid_loader)}")

    
def evaluate(model, test_loader):            
    # Testing
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            output = model(
                batch['input_ids'].squeeze().to(CONFIG['device']),
                attention_mask = batch['mask'].to(CONFIG['device']),
                labels = batch['target'].to(CONFIG['device'])
            )
            test_acc += (output['logits'].argmax(1) == batch['target'].to(CONFIG['device'])).sum().item()

            torch.cuda.empty_cache()
    print(f"Test Acc: {test_acc / len(test_loader)}")
