import os
import json
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from dotenv import load_dotenv
import torch

# 환경 변수 로드
load_dotenv()

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StarCoderTrainingAPIView(APIView):
    """
    StarCoder 모델 Fine-Tuning API
    """

    model_name = "bigcode/starcoder"
    tokenizer = None
    model = None

    @classmethod
    def initialize_model(cls):
        """
        StarCoder 모델 및 토크나이저 초기화
        """
        try:
            api_key = os.getenv("HUGGINGFACE_API_KEY", "")
            if not api_key:
                raise ValueError("Hugging Face API key is missing.")
            
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, token=api_key)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, token=api_key)

            if cls.tokenizer.pad_token is None:
                cls.tokenizer.add_special_tokens({'pad_token': cls.tokenizer.eos_token})
                cls.model.resize_token_embeddings(len(cls.tokenizer))

            logger.info("StarCoder model and tokenizer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {e}")

    def load_dataset(self, jsonl_path):
        """
        JSONL 파일 로드 및 데이터셋 생성
        """
        try:
            with open(jsonl_path, "r", encoding="utf-8") as file:
                data = [json.loads(line) for line in file]

            # 데이터셋의 무효 항목 필터링
            valid_data = [entry for entry in data if entry.get("prompt") and entry.get("completion")]
            if not valid_data:
                raise ValueError("No valid entries found in dataset.")

            logger.info(f"Loaded dataset with {len(valid_data)} valid entries from {jsonl_path}")
            return Dataset.from_dict({
                "prompt": [entry["prompt"] for entry in valid_data],
                "completion": [entry["completion"] for entry in valid_data]
            })
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise RuntimeError(f"Error loading dataset: {e}")

    def preprocess_and_tokenize(self, dataset, tokenizer, max_length=2048):
        """
        데이터셋 전처리 및 토크나이징
        """
        def tokenize_function(examples):
            combined_text = examples["prompt"] + examples["completion"]
            try:
                tokens = tokenizer(
                    combined_text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                )
                tokens["labels"] = tokens["input_ids"].copy()
                return tokens
            except Exception as e:
                logger.error(f"Error tokenizing example: {examples}")
                raise RuntimeError(f"Tokenization error: {e}")

        try:
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            logger.info(f"Tokenized dataset size: {len(tokenized_dataset)} samples")
            return tokenized_dataset
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise RuntimeError(f"Error during tokenization: {e}")

    def train_starcoder_model(self, train_dataset, eval_dataset, tokenizer, output_dir):
        """
        StarCoder 모델 Fine-Tuning
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=200,
                weight_decay=0.01,
                learning_rate=2e-5,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=True,
                report_to="none"
            )

            data_collator = DataCollatorWithPadding(tokenizer)
            trainer = Trainer(
                model=self.__class__.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )

            trainer.train()
            self.__class__.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Fine-tuned model saved at {output_dir}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise RuntimeError(f"Error during training: {e}")

    def post(self, request):
        jsonl_path = "./starcoder_input.jsonl"
        output_dir = "./fine_tuned_starcoder_model"

        if not self.__class__.tokenizer or not self.__class__.model:
            try:
                self.__class__.initialize_model()
            except RuntimeError as e:
                return Response({"error": str(e)}, status=500)

        try:
            # 데이터셋 로드 및 토크나이징
            dataset = self.load_dataset(jsonl_path)
            tokenized_dataset = self.preprocess_and_tokenize(dataset, self.__class__.tokenizer)

            # 데이터셋 길이 확인
            logger.info(f"Total tokenized dataset length: {len(tokenized_dataset)}")

            # 학습 및 평가 데이터 분리
            split_data = tokenized_dataset.train_test_split(test_size=0.2)
            train_dataset = split_data["train"]
            eval_dataset = split_data["test"]

            logger.info(f"Train dataset length: {len(train_dataset)}, Eval dataset length: {len(eval_dataset)}")

            # 모델 학습
            self.train_starcoder_model(train_dataset, eval_dataset, self.__class__.tokenizer, output_dir)
        except Exception as e:
            logger.error(f"Error during preprocessing or training: {str(e)}")
            return Response({"error": f"Error during preprocessing or training: {str(e)}"}, status=500)

        return Response({"message": "Model Fine-Tuning completed successfully.", "output_dir": output_dir})
