from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Hugging Face에서 모델 & 토크나이저 다운로드
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# 원하는 경로에 저장 (예: ./local/mbart)
model.save_pretrained("./local/mbart")
tokenizer.save_pretrained("./local/mbart")