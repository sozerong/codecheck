import json
import os
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from dotenv import load_dotenv  # 환경 변수 로드

load_dotenv()

class Llama3TrainingAPIView(APIView):
    """
    Llama3 모델 학습 및 저장
    """
    @swagger_auto_schema(
        operation_description="Llama3 모델 학습 수행",
        responses={200: openapi.Response("성공 메시지와 모델 저장 경로")}
    )
    def post(self, request):
        # JSON 파일 및 모델 설정
        json_path = "./llama3_input.json"
        model_name = "beomi/Llama-3-Open-Ko-8B"
        output_dir = "./custom_llama3_model"

        # 환경 변수에서 Hugging Face API 키 로드
        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if not api_key:
            return Response({"error": "Hugging Face API key is missing. Please set HUGGINGFACE_API_KEY."}, status=500)

        # 데이터 로드 함수
        def load_training_data(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                print(f"Training data loaded successfully: {len(data)} entries")
                return Dataset.from_dict({
                    "instruction": [d["instruction"] for d in data],
                    "input": [d["input"] for d in data],
                    "output": [d["output"] for d in data]
                })
            except FileNotFoundError:
                return Response({"error": f"Input JSON file '{json_path}' not found."}, status=404)
            except Exception as e:
                return Response({"error": f"Failed to load training data: {str(e)}"}, status=500)

        # 모델 학습 함수
        def train_llama3_model(dataset, model_name, output_dir):
            try:
                # 모델과 토크나이저 로드
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
                model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key)

                # 데이터 토크나이징
                tokenized_data = dataset.map(
                    lambda x: tokenizer(
                        x["instruction"] + "\n" + x["input"] + "\n" + x["output"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    ),
                    batched=True
                )

                # 학습 설정
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    save_steps=10,
                    save_total_limit=2,
                    logging_dir="./logs",
                    evaluation_strategy="epoch"
                )

                # 트레이너 초기화
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_data
                )

                # 학습 실행
                trainer.train()

                # 모델 저장
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Model saved at {output_dir}")

            except Exception as e:
                print(f"Error during training: {str(e)}")
                return Response({"error": f"Failed to train model: {str(e)}"}, status=500)

        # 학습 데이터 로드
        dataset = load_training_data(json_path)
        if isinstance(dataset, Response):  # 오류 발생 시 Response 반환
            return dataset

        # 모델 학습
        train_llama3_model(dataset, model_name, output_dir)

        return Response({"message": "Llama3 model trained and saved successfully", "output_dir": output_dir})
