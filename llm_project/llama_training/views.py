import os
import json
import logging
import boto3
from botocore.exceptions import ClientError
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_aws_parameter(parameter_name, with_decryption=True):
    """
    AWS Parameter Store에서 환경 변수를 가져오는 함수
    """
    try:
        # AWS SSM 클라이언트 생성
        ssm_client = boto3.client('ssm', region_name=os.getenv("AWS_REGION", "us-east-1"))

        # Parameter 가져오기
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=with_decryption)
        return response['Parameter']['Value']
    except ClientError as e:
        logger.error(f"Error fetching parameter {parameter_name}: {str(e)}")
        raise e

class Llama3TrainingAPIView(APIView):
    """
    Llama3 모델 학습 및 데이터셋 생성
    """

    # 클래스 변수로 모델과 토크나이저 초기화
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = None
    model = None

    @classmethod
    def initialize_model(cls):
        """
        모델과 토크나이저를 초기화하는 메서드
        """
        try:
            # AWS Parameter Store에서 Hugging Face API Key 가져오기
            api_key = get_aws_parameter("CodeCheck")
            if not api_key:
                raise ValueError("Hugging Face API key is missing.")

            # 모델 및 토크나이저 로드
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, token=api_key)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, token=api_key)

            # 패딩 토큰 추가
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.add_special_tokens({'pad_token': cls.tokenizer.eos_token})
                cls.model.resize_token_embeddings(len(cls.tokenizer))  # 모델에 새 토큰 반영

            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    def preprocess_and_tokenize(self, dataset, tokenizer, max_length=4096):
        """
        데이터셋 전처리 및 토크나이징
        """
        def tokenize_function(examples):
            combined_text = examples["instruction"] + "\n" + examples["input"]
            tokens = tokenizer(
                combined_text,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized_data = dataset.map(tokenize_function, batched=False)
        logger.info(f"Before filtering: {len(tokenized_data)} samples")

        # 초과 길이 데이터 필터링
        tokenized_data = tokenized_data.filter(
            lambda x: len(x["input_ids"]) <= max_length
        )
        logger.info(f"After filtering: {len(tokenized_data)} samples")

        if len(tokenized_data) == 0:
            raise ValueError("No valid samples found after tokenization.")

        tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_data

    def train_llama3_model(self, train_dataset, eval_dataset, tokenizer, output_dir):
        """
        모델 학습 및 저장
        """
        try:
            model = self.__class__.model
            if not model:
                raise ValueError("Model is not initialized.")

            # 학습 설정
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=10,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                warmup_steps=200,
                weight_decay=0.01,
                learning_rate=3e-5,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                save_total_limit=3,
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=True,
                report_to="none",
            )

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            # Trainer 생성
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            # 학습 실행
            trainer.train()

            # 모델 저장
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved at {output_dir}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise RuntimeError(f"Error during training: {e}")

    def post(self, request):
        """
        POST 요청으로 학습 실행
        """
        json_path = "./llama3_input.json"
        output_dir = "./custom_llama3_model"

        # 모델과 토크나이저 초기화
        if not self.__class__.tokenizer or not self.__class__.model:
            try:
                self.__class__.initialize_model()
            except ValueError as e:
                return Response({"error": str(e)}, status=500)

        # 입력 JSON 파일 로드
        try:
            with open(json_path, "r", encoding="utf-8") as input_file:
                data = json.load(input_file)
            logger.info(f"Input JSON loaded successfully with {len(data)} entries.")
        except Exception as e:
            logger.error(f"Failed to load input JSON: {str(e)}")
            return Response({"error": f"Failed to load input JSON: {str(e)}"}, status=500)

        # 데이터셋 생성
        try:
            dataset = Dataset.from_dict({
                "instruction": [d["instruction"] for d in data],
                "input": [d["input"] for d in data],
                "output": [d["output"] for d in data],
            })
            tokenized_data = self.preprocess_and_tokenize(dataset, self.__class__.tokenizer)

            # 학습 및 평가 데이터 분리
            split_data = tokenized_data.train_test_split(test_size=0.2)
            train_dataset = split_data["train"]
            eval_dataset = split_data["test"]

            # 모델 학습
            self.train_llama3_model(train_dataset, eval_dataset, self.__class__.tokenizer, output_dir)
        except Exception as e:
            logger.error(f"Error during preprocessing or training: {str(e)}")
            return Response({"error": f"Error during preprocessing or training: {str(e)}"}, status=500)

        return Response({"message": "Model training completed successfully.", "output_dir": output_dir})
