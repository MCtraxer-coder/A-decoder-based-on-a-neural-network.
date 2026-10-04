import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tokinizer_and_dataset import create_dataloaders, CharTokenizer
import json

class CipherClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, max_len=50, num_classes=3):
        super(CipherClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

class CipherDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_len=50, num_ciphers=3):
        super(CipherDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cipher_embedding = nn.Embedding(num_ciphers, d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x, cipher_idx):
        x = self.embedding(x) * np.sqrt(self.d_model)
        cipher_emb = self.cipher_embedding(cipher_idx)
        cipher_emb = cipher_emb.unsqueeze(1)
        x = x + cipher_emb
        x = x + self.pe[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

def train_classifier(model, train_loader, val_loader, tokenizer, epochs=15, lr=0.001, device='cpu', save_path='classifier_best.pth'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("=" * 60)
    print(" ОБУЧЕНИЕ КЛАССИФИКАТОРА")
    print("=" * 60 + "\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x = batch['encoded'].to(device)
            y = batch['cipher_idx'].to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['encoded'].to(device)
                y = batch['cipher_idx'].to(device)

                output = model(x)
                loss = criterion(output, y)

                val_loss += loss.item()
                preds = output.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" Классификатор сохранён (val_acc: {val_acc:.2f}%)")

        print(f"\n Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"                        | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("-" * 60)

    print(f"\n Классификатор обучен! Лучшая точность: {best_val_acc:.2f}%")
    return model, history

def train_decoder(decoder, classifier, train_loader, val_loader, tokenizer, epochs=50, lr=0.001, device='cpu', save_path='decoder_best.pth', freeze_classifier=True):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    decoder.to(device)
    classifier.to(device)

    if freeze_classifier:
        if os.path.exists('classifier_best.pth'):
            classifier.load_state_dict(torch.load('classifier_best.pth'))
            print(" Классификатор загружен из classifier_best.pth")
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        print(" Классификатор заморожен (веса не обновляются)")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    vocab_size = tokenizer.vocab_size

    print("\n" + "=" * 60)
    print(" ОБУЧЕНИЕ ДЕШИФРАТОРА (с подсказкой от классификатора)")
    print("=" * 60 + "\n")

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            x = batch['encoded'].to(device)
            y = batch['original'].to(device)
            cipher_idx = batch['cipher_idx'].to(device)
            mask = (y != tokenizer.pad_idx)

            optimizer.zero_grad()
            output = decoder(x, cipher_idx)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=-1)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100 if total > 0 else 0

        decoder.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch['encoded'].to(device)
                y = batch['original'].to(device)
                mask = (y != tokenizer.pad_idx)

                logits = classifier(x)
                cipher_idx = logits.argmax(dim=1)

                output = decoder(x, cipher_idx)
                loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))

                val_loss += loss.item()
                preds = output.argmax(dim=-1)
                val_correct += ((preds == y) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(decoder.state_dict(), save_path)
            print(f" Дешифратор сохранён (val_acc: {val_acc:.2f}%)")

        print(f"\n Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"                        | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("-" * 60)

    print(f"\n Дешифратор обучен! Лучший val_loss: {best_val_acc:.4f}")
    return decoder, history

class CascadeCipherModel:
    def __init__(self, classifier, decoder, tokenizer, device='cpu'):
        self.classifier = classifier
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.classifier.to(self.device)
        self.decoder.to(self.device)
        self.classifier.eval()
        self.decoder.eval()

        self.idx2cipher = {0: 'caesar', 1: 'atbash', 2: 'vigenere'}

    def decrypt(self, text, max_len=50):
        encoded = self.tokenizer.encode(text, max_len=max_len)
        padded = self.tokenizer.pad_sequence(encoded, max_len)
        x = torch.tensor([padded], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.classifier(x)
            cipher_idx = logits.argmax(dim=1).item()
            cipher_name = self.idx2cipher[cipher_idx]

        with torch.no_grad():
            output = self.decoder(x, torch.tensor([cipher_idx]).to(self.device))
            preds = output.argmax(dim=-1)

        text_len = len(text)
        decrypted = self.tokenizer.decode(preds[0][:text_len].tolist())
        return decrypted, cipher_name

if __name__ == "__main__":
    print(" ТЕСТОВЫЙ ЗАПУСК model_train.py")
    print("=" * 60)

    train_loader, val_loader, tokenizer = create_dataloaders(
        file_path="dataset.jsonl",
        max_len=50,
        batch_size=128,
        train_split=0.8,
        max_samples=5000,
        apply_noise_in_dataset=True,
        noise_level=0.1,
        gap_level=0.1
    )

    vocab_size = tokenizer.vocab_size

    classifier = CipherClassifier(vocab_size)
    classifier, hist_cls = train_classifier(
        classifier, train_loader, val_loader, tokenizer,
        epochs=10,
        lr=0.001,
        device='cpu'
    )

    decoder = CipherDecoder(vocab_size)
    decoder, hist_dec = train_decoder(
        decoder, classifier, train_loader, val_loader, tokenizer,
        epochs=20,
        lr=0.001,
        device='cpu',
        freeze_classifier=True
    )

    cascade = CascadeCipherModel(classifier, decoder, tokenizer)

    test_texts = [
        "сурьёу",
        "отъдт",
        "ыуацй"
    ]

    print("\n" + "=" * 60)
    print(" ТЕСТ КАСКАДНОЙ МОДЕЛИ")
    print("=" * 60)

    for text in test_texts:
        decrypted, cipher = cascade.decrypt(text)
        print(f"\n  {text} → {decrypted} (определено: {cipher})")