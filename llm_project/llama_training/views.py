import json
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

class Llama3TrainingAPIView(APIView):
    """
    Llama3 모델 학습 및 저장
    """
    @swagger_auto_schema(
        operation_description="Llama3 모델 학습 수행",
        responses={200: openapi.Response('성공 메시지와 모델 저장 경로')}
    )
    def post(self, request):
        json_path = "./llama3_input.json"
        model_name = "beomi/Llama-3-Open-Ko-8B"
        output_dir = "./custom_llama3_model"

        def load_training_data(json_path):
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return Dataset.from_dict({
                "instruction": [d["instruction"] for d in data],
                "input": [d["input"] for d in data],
                "output": [d["output"] for d in data]
            })

        def train_llama3_model(dataset, model_name, output_dir):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            tokenized_data = dataset.map(lambda x: tokenizer(x["instruction"] + "\n" + x["input"] + "\n" + x["output"], truncation=True, padding="max_length"), batched=True)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                save_steps=10,
                save_total_limit=2,
                logging_dir="./logs",
                evaluation_strategy="epoch"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_data
            )
            
            trainer.train()
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved at {output_dir}")

        dataset = load_training_data(json_path)
        train_llama3_model(dataset, model_name, output_dir)

        return Response({"message": "Llama3 model trained and saved successfully", "output_dir": output_dir})
