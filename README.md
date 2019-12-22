# KoBERT-nsmc

- Naver-movie-review sentiment classification with KoBERT
- Implement with `Huggingface Tranformers` library

## Dependencies

- torch>=1.1.0
- transformers>=2.2.2
- sentencepiece>=0.1.82
- scikit-learn

## How to use KoBERT on Huggingface Transformers Library

- From transformers v2.2.2, you can upload/download personal bert model directly.
- To use tokenizer, you have to import `KoBertTokenizer` from `tokenization_kobert.py`.

```python
from transformers import BertModel
from tokenization_kobert import KoBertTokenizer

model = BertModel.from_pretrained('monologg/kobert')
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
```

## Usage

```bash
# 1. Download data
$ cd data
$ ./download_data.sh

# 2. Train model and eval
$ cd ..
$ python3 main.py --model_type kobert --do_train --do_eval
```

## Results

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | **89.63**    |
| DistilKoBERT      | 88.28        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC dataset](https://github.com/e9t/nsmc)
