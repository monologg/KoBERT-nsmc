# KoBERT-nsmc

- KoBERT를 이용한 네이버 영화 리뷰 감정 분석 (sentiment classification)
- `Huggingface Tranformers` 라이브러리를 이용하여 구현

## Dependencies

- torch>=1.1.0
- transformers>=2.2.2
- sentencepiece>=0.1.82
- scikit-learn

## How to use KoBERT on Huggingface Transformers Library

- 기존의 KoBERT를 transformers 라이브러리에서 곧바로 사용할 수 있도록 맞췄습니다.
  - transformers v2.2.2부터 개인이 만든 모델을 transformers를 통해 직접 업로드/다운로드하여 사용할 수 있습니다
- Tokenizer를 사용하려면 `tokenization_kobert.py`에서 `KoBertTokenizer`를 임포트해야 합니다.

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
| DistilKoBERT      | 88.39        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC dataset](https://github.com/e9t/nsmc)
