import os
import re
import json
import random
import pandas as pd

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
ALPHABET_UPPER = ALPHABET.upper()
PUNCTUATION = " .,!?-"
ALL_CHARS = ALPHABET + ALPHABET_UPPER + PUNCTUATION

HF_PATH = "hf://datasets/capapdsa/russian-instructions-10k/russian-instructions-10k.jsonl"

SOURCE_FILE = "source_text.txt"
OUTPUT_FILE = "dataset.jsonl"

DATASET_SIZE = 50000
MIN_LEN = 10
MAX_LEN = 50
UNIQUE_ONLY = True

def load_hf_texts():
    print(f" Скачиваем датасет: {HF_PATH}")
    df = pd.read_json(HF_PATH, lines=True)
    print(f" Загружено {len(df)} строк. Колонки: {list(df.columns)}")

    texts = []
    for col in df.columns:
        if df[col].dtype == object:
            for val in df[col].dropna():
                if isinstance(val, str) and len(val.strip()) > 0:
                    texts.append(val.strip())

    print(f" Собрано {len(texts)} текстовых фрагментов")
    return texts


def split_into_sentences(texts):

    sentences = []
    for text in texts:
        text = re.sub(r'\s+', ' ', text)
        parts = re.split(r'(?<=[.!?])\s+', text)
        for p in parts:
            p = p.strip()
            if not re.search(r'[а-яА-ЯёЁ]', p):
                continue
            if MIN_LEN <= len(p) <= MAX_LEN * 2:
                sentences.append(p)

    if UNIQUE_ONLY:
        sentences = list(set(sentences))

    print(f" Получено {len(sentences)} предложений после фильтрации")
    return sentences


def save_source_text(sentences, path=SOURCE_FILE):
    with open(path, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
    print(f" source_text.txt сохранён ({len(sentences)} строк)")


def load_source_text(path=SOURCE_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} не найден. Сначала запусти make_dataset.py")
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def smart_truncate(text, max_len=MAX_LEN):

    text = text.strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]

    if truncated[-1] not in " .,!?-":
        truncated = truncated.rsplit(' ', 1)[0]
    return truncated.strip()

def caesar(text, shift=3):
    result = []
    for ch in text:
        if ch in ALPHABET:
            idx = (ALPHABET.index(ch) + shift) % len(ALPHABET)
            result.append(ALPHABET[idx])
        elif ch in ALPHABET_UPPER:
            idx = (ALPHABET_UPPER.index(ch) + shift) % len(ALPHABET_UPPER)
            result.append(ALPHABET_UPPER[idx])
        else:
            result.append(ch)
    return ''.join(result)


def atbash(text):
    result = []
    for ch in text:
        if ch in ALPHABET:
            idx = len(ALPHABET) - 1 - ALPHABET.index(ch)
            result.append(ALPHABET[idx])
        elif ch in ALPHABET_UPPER:
            idx = len(ALPHABET_UPPER) - 1 - ALPHABET_UPPER.index(ch)
            result.append(ALPHABET_UPPER[idx])
        else:
            result.append(ch)
    return ''.join(result)


def vigenere(text, keyword="ключ"):
    result = []
    keyword = keyword.lower()
    key_idx = 0
    for ch in text:
        if ch in ALPHABET:
            shift = ALPHABET.index(keyword[key_idx % len(keyword)])
            idx = (ALPHABET.index(ch) + shift) % len(ALPHABET)
            result.append(ALPHABET[idx])
            key_idx += 1
        elif ch in ALPHABET_UPPER:
            shift = ALPHABET.index(keyword[key_idx % len(keyword)])
            idx = (ALPHABET_UPPER.index(ch) + shift) % len(ALPHABET_UPPER)
            result.append(ALPHABET_UPPER[idx])
            key_idx += 1
        else:
            result.append(ch)
    return ''.join(result)

def generate_dataset(
    cipher_types=None,
    size=DATASET_SIZE,
    output_file=OUTPUT_FILE,
):
    if cipher_types is None:
        cipher_types = ['caesar', 'atbash', 'vigenere']

    if not os.path.exists(SOURCE_FILE):
        print(f"⚠️ {SOURCE_FILE} не найден. Скачиваю датасет с HF...")
        texts = load_hf_texts()
        sentences = split_into_sentences(texts)
        save_source_text(sentences)
    else:
        sentences = load_source_text()
        print(f" Загружено {len(sentences)} предложений из {SOURCE_FILE}")

    if len(sentences) == 0:
        print(" Нет предложений. Проверь source_text.txt")
        return

    print(f"\n Генерирую {size} примеров...")
    print(f"   MAX_LEN: {MAX_LEN}")
    print(f"   Шум: ВЫКЛ")
    print(f"   Пропуски: ВЫКЛ")

    with open(output_file, 'w', encoding='utf-8') as f:
        created = 0
        attempts = 0
        max_attempts = size * 5

        while created < size and attempts < max_attempts:
            attempts += 1
            original = smart_truncate(random.choice(sentences), MAX_LEN)

            if len(original) < MIN_LEN:
                continue

            cipher_type = random.choice(cipher_types)

            if cipher_type == 'caesar':
                shift = random.randint(1, 25)
                encoded = caesar(original, shift)
                params = {'shift': shift}
            elif cipher_type == 'atbash':
                encoded = atbash(original)
                params = {}
            else:
                keyword = random.choice(['ключ', 'пароль', 'шифр', 'код', 'тайна'])
                encoded = vigenere(original, keyword)
                params = {'keyword': keyword}

            record = {
                'original': original,
                'clean_encoded': encoded,
                'cipher': cipher_type,
                'params': params,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            created += 1

            if created % 5000 == 0:
                print(f"  Создано {created}/{size}")

    print(f" dataset.jsonl создан! {created} примеров")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_FILE):
        print("=" * 60)
        print("ШАГ 1: Скачивание и подготовка текста с HuggingFace")
        print("=" * 60)
        texts = load_hf_texts()
        sentences = split_into_sentences(texts)
        if len(sentences) < 100:
            print(f" Мало предложений ({len(sentences)}). Проверь датасет.")
        save_source_text(sentences)

    print("\n" + "=" * 60)
    print("ШАГ 2: Генерация dataset.jsonl")
    print("=" * 60)
    generate_dataset()