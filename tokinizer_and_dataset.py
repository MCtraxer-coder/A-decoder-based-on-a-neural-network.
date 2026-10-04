import torch
import json
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter

def load_dataset(file_path="dataset.jsonl", max_samples=None):
    data=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data

class CharTokenizer:
    def __init__(self, texts, special_tokens=['<PAD>','<UNK>']):
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        all_chars = sorted(all_chars)
        self.cx = {token: idx for idx, token in enumerate(special_tokens)}
        for idx, char in enumerate(all_chars, start=len(special_tokens)):
            self.cx[char] = idx
        self.ir = {idx: char for char, idx in self.cx.items()}
        self.pad_idx = self.cx['<PAD>']
        self.unk_idx = self.cx['<UNK>']
        self.vocab_size = len(self.cx)
        print(f" Токенизатор создан. Размер словаря: {self.vocab_size}")
        print(f"   Спецтокены: <PAD>={self.pad_idx}, <UNK>={self.unk_idx}")

    def encode(self, text, max_len=None):
        indices=[]
        for ch in text:
            if ch in self.cx:
                indices.append(self.cx[ch])
            else:
                indices.append(self.unk_idx)
        if max_len:
            indices = indices[:max_len]
        return indices
    def decode(self, indices, ignore_pad=True):
        chars =[]
        for idx in indices:
            if ignore_pad and idx == self.pad_idx:
                continue
            if idx in self.ir:
                chars.append(self.ir[idx])
            else:
                chars.append('<UNK>')
        return ''.join(chars)
    def pad_sequence(self, indices, max_len, pad_value=None):
        if pad_value is None:
            pad_value = self.pad_idx
        if len(indices) >= max_len:
            return indices[:max_len]
        else:
            return indices + [pad_value] *(max_len - len(indices))

def apply_noise(text, noise_level=0.1):
    Alph = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    Alph_upper = Alph.upper()
    text = list(text)
    for i, ch in enumerate(text):
        if random.random() < noise_level:
            if ch in Alph:
                text[i] = random.choice(Alph)
            elif ch in Alph_upper:
                text[i] = random.choice(Alph_upper)
    return ''.join(text)
def apply_gaps(text, gaps_level=0.1):
    Alph = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    Alph_upper = Alph.upper()
    text = list(text)
    result = []
    for ch in text:
        if ch in Alph or ch in Alph_upper:
            if random.random() < gaps_level:
                continue
        result.append(ch)
    return ''.join(result)

class Cipherdataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50, apply_noise_in_dataset=False, noise_level=0.0, gap_level=0.0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.apply_noise_in_dataset = apply_noise_in_dataset
        self.noise_level = noise_level
        self.gap_level = gap_level
        self.cipher2idx = {
            'caesar': 0,
            'atbash': 1,
            'vigenere': 2
        }
        print(f"Датасет загружен: {len(data)} примеров")
        if apply_noise_in_dataset:
            print(f" Применяем шум: {noise_level*100:.0f}%, пропуски:{gap_level*100:.0f}%")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        record = self.data[idx]
        encoded_text = record["clean_encoded"]
        if self.apply_noise_in_dataset:
            if self.noise_level > 0:
                encoded_text = apply_noise(encoded_text, self.noise_level)
            if self.gap_level > 0:
                encoded_text = apply_gaps(encoded_text, self.gap_level)
        es = self.tokenizer.encode(encoded_text, max_len=self.max_len)
        es = self.tokenizer.pad_sequence(es, self.max_len)
        os = self.tokenizer.encode(record['original'], max_len=self.max_len)
        os = self.tokenizer.pad_sequence(os, self.max_len)
        cipher_type = record['cipher']
        cipher_idx = self.cipher2idx[cipher_type]

        return {
            'encoded': torch.tensor(es, dtype=torch.long),
            'original': torch.tensor(os, dtype=torch.long),
            'cipher_idx': torch.tensor(cipher_idx, dtype=torch.long),
            'cipher_name': cipher_type,
            'clean_encoded': record['clean_encoded']
        }

def create_dataloaders(file_path="dataset.jsonl",
                       max_len =50,
                       batch_size=64,
                       train_split=0.8,
                       max_samples=None,
                       apply_noise_in_dataset = False,
                       noise_level = 0.1,
                       gap_level = 0.1

):
    data = load_dataset(file_path, max_samples)
    all_texts =[]
    for record in data:
        all_texts.append(record['clean_encoded'])
        all_texts.append(record['original'])
    tokenizer = CharTokenizer(all_texts)
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    train_dataset = Cipherdataset(train_data, tokenizer, max_len, apply_noise_in_dataset=apply_noise_in_dataset, noise_level=noise_level, gap_level=gap_level)
    val_dataset = Cipherdataset(val_data, tokenizer, max_len, apply_noise_in_dataset=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle= False)
    print(f" Train: {len(train_dataset)} примеров, Val: {len(val_dataset)} примеров")
    print(f"   Размер батча: {batch_size}, Макс. длина: {max_len}")
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, tokenizer = create_dataloaders(
        file_path="dataset.jsonl",
        max_len=50,
        batch_size=32,
        max_samples=500,
        apply_noise_in_dataset=True,
        noise_level=0.1,
        gap_level=0.1
    )

    batch = next(iter(train_loader))
    print(f"\nПример батча:")
    print(f"  encoded shape: {batch['encoded'].shape}")
    print(f"  original shape: {batch['original'].shape}")
    print(f"  cipher_idx: {batch['cipher_idx']}")
    print(f"  cipher_name: {batch['cipher_name']}")



