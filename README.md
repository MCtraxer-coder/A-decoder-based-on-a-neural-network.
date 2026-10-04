# Cipher Decoder — каскадная нейросеть для расшифровки

Определяет тип шифра (Цезарь / Атбаш / Виженер) и расшифровывает русский текст.

## Архитектура

- **Классификатор** — TransformerEncoder, определяет тип шифра (точность ~97%)
- **Дешифратор** — TransformerEncoder + embedding типа шифра, восстанавливает оригинал
- **Каскад** — классификатор → дешифратор

## Установка

```bash
git clone https://github.com/MCtraxer-coder/cipher_project.git
cd cipher_project
pip install -r requirements.txt