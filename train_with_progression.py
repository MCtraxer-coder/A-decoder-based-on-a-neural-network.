import torch
import json
import os
from model_train import (
    CipherClassifier,
    CipherDecoder,
    train_classifier,
    train_decoder,
    CascadeCipherModel
)
from tokinizer_and_dataset import create_dataloaders

FORCE_RETRAIN = False
TRAIN_DECODER = True
DATASET_SIZE = 50000
BATCH_SIZE = 128
CLASSIFIER_EPOCHS = 30
DECODER_EPOCHS = 40
NOISE_LEVEL = 0.0
GAP_LEVEL = 0.0
MAX_LEN = 50
DEVICE = 'cpu'


def save_cascade(classifier, decoder, tokenizer, filepath='cascade_full.pth'):
    checkpoint = {
        'classifier_state': classifier.state_dict(),
        'decoder_state': decoder.state_dict(),
        'cipher2idx': {'caesar': 0, 'atbash': 1, 'vigenere': 2},
        'vocab_size': tokenizer.vocab_size,
        'pad_idx': tokenizer.pad_idx
    }
    torch.save(checkpoint, filepath)
    print(f" Каскад сохранён в {filepath}")


def progressive_training():
    print("=" * 60)
    print("   КАСКАДНОЕ ОБУЧЕНИЕ")
    print(f"   Режим: {'ПРИНУДИТЕЛЬНОЕ ОБУЧЕНИЕ' if FORCE_RETRAIN else 'ЗАГРУЗКА / ДООБУЧЕНИЕ'}")
    print(f"   Дешифратор: {'ВКЛЮЧЁН' if TRAIN_DECODER else 'ВЫКЛЮЧЁН'}")
    print(f"   Датасет: {DATASET_SIZE} примеров")
    print(f"   Эпохи классификатора: {CLASSIFIER_EPOCHS if FORCE_RETRAIN else '—'}")
    print(f"   Эпохи дешифратора: {DECODER_EPOCHS if TRAIN_DECODER else '—'}")
    print("=" * 60 + "\n")


    print(' Загружаем данные из dataset.jsonl...')
    train_loader, val_loader, tokenizer = create_dataloaders(
        file_path='dataset.jsonl',
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        max_samples=DATASET_SIZE,
        apply_noise_in_dataset=(NOISE_LEVEL > 0 or GAP_LEVEL > 0),
        noise_level=NOISE_LEVEL,
        gap_level=GAP_LEVEL
    )

    vocab_size = tokenizer.vocab_size
    print(f"  Размер словаря: {vocab_size}\n")

    if FORCE_RETRAIN:
        print(" ОБУЧЕНИЕ КЛАССИФИКАТОРА С НУЛЯ\n")
        print("\n" + "=" * 60)
        print(" ЭТАП 1: ОБУЧЕНИЕ КЛАССИФИКАТОРА")
        print("=" * 60 + "\n")

        classifier = CipherClassifier(vocab_size, max_len=MAX_LEN).to(DEVICE)
        classifier, hist_classifier = train_classifier(
            model=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            epochs=CLASSIFIER_EPOCHS,
            lr=0.001,
            device=DEVICE,
            save_path='classifier_best.pth'
        )

        with open('history_classifier.json', 'w', encoding='utf-8') as f:
            json.dump(hist_classifier, f, ensure_ascii=False, indent=2)
        print(" История классификатора сохранена в history_classifier.json")

    else:
        print(" ЗАГРУЗКА КЛАССИФИКАТОРА (FORCE_RETRAIN = False)")

        if os.path.exists('classifier_best.pth'):
            classifier = CipherClassifier(vocab_size, max_len=MAX_LEN).to(DEVICE)
            classifier.load_state_dict(torch.load('classifier_best.pth', map_location=DEVICE))
            classifier.eval()
            print(" Классификатор загружен из classifier_best.pth\n")
        else:
            print("️ classifier_best.pth не найден. Обучаю с нуля...\n")
            classifier = CipherClassifier(vocab_size, max_len=MAX_LEN).to(DEVICE)
            classifier, hist_classifier = train_classifier(
                model=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                epochs=CLASSIFIER_EPOCHS,
                lr=0.001,
                device=DEVICE,
                save_path='classifier_best.pth'
            )
            with open('history_classifier.json', 'w', encoding='utf-8') as f:
                json.dump(hist_classifier, f, ensure_ascii=False, indent=2)

    if TRAIN_DECODER:
        print("\n" + "=" * 60)
        print(" ЭТАП 2: ОБУЧЕНИЕ / ДООБУЧЕНИЕ ДЕШИФРАТОРА")
        print("=" * 60 + "\n")

        decoder = CipherDecoder(vocab_size, max_len=MAX_LEN).to(DEVICE)

        if os.path.exists('decoder_best.pth'):
            decoder.load_state_dict(torch.load('decoder_best.pth', map_location=DEVICE))
            print(" Дешифратор загружен из decoder_best.pth (ДООБУЧЕНИЕ)")
            print(f"   Продолжаем с текущих весов на {DECODER_EPOCHS} эпох\n")
        else:
            print(" Дешифратор новый (обучение с нуля)\n")

        decoder, hist_decoder = train_decoder(
            decoder=decoder,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            epochs=DECODER_EPOCHS,
            lr=0.001,
            device=DEVICE,
            save_path='decoder_best.pth',
            freeze_classifier=True
        )

        with open('history_decoder.json', 'w', encoding='utf-8') as f:
            json.dump(hist_decoder, f, ensure_ascii=False, indent=2)
        print(" История дешифратора сохранена в history_decoder.json")

        save_cascade(classifier, decoder, tokenizer, 'cascade_full.pth')

    else:
        print("\n Дешифратор пропущен (TRAIN_DECODER = False)")
        if os.path.exists('decoder_best.pth'):
            decoder = CipherDecoder(vocab_size, max_len=MAX_LEN).to(DEVICE)
            decoder.load_state_dict(torch.load('decoder_best.pth', map_location=DEVICE))
            decoder.eval()
            print(" Дешифратор загружен для теста")
        else:
            decoder = None
            print(" decoder_best.pth не найден")

    if decoder is not None:
        print("\n" + "=" * 60)
        print("    ТЕСТИРОВАНИЕ КАСКАДА")
        print("=" * 60 + "\n")

        cascade = CascadeCipherModel(classifier, decoder, tokenizer, DEVICE)

        test_examples = [
            "Юбкнэйш, чгэ юьъох пчкъчхшщ",
            "Фхоюо монув Ёскрчёуйцё",
            "Лкж: Съьфё Пфкпп п Иглщфжуж",
        ]

        print("Зашифровано → Расшифровано (тип шифра):")
        print("-" * 40)

        for text in test_examples:
            decrypted, cipher = cascade.decrypt(text)
            print(f"  {text} → {decrypted} (тип: {cipher})")
    else:
        print("\n Тестирование каскада пропущено (дешифратор не доступен)")

    print("\n" + "=" * 60)
    print(" ГОТОВО!")
    print("=" * 60)


if __name__ == "__main__":
    progressive_training()