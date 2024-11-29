import os
import json
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AIEvaluationAPIView(APIView):
    """
    Llama 모델을 사용하여 코드를 평가하고 결과를 저장하는 API
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
        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if not api_key:
            raise ValueError("Hugging Face API key is missing.")
        try:
            # 모델 및 토크나이저 로드
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_auth_token=api_key)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, use_auth_token=api_key)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    @swagger_auto_schema(
        operation_description="Llama3 모델을 사용하여 코드를 평가하고 JSON 파일로 저장합니다.",
        responses={200: openapi.Response("AI 평가 완료 메시지와 결과 JSON 경로")}
    )
    def post(self, request):
        """
        POST 요청으로 평가를 실행하고 결과를 JSON 파일로 저장
        """
        input_json_path = "./llama3_input.json"
        output_json_path = "./llama3_output.json"

        # 모델과 토크나이저 로드 확인
        if not self.__class__.tokenizer or not self.__class__.model:
            try:
                self.__class__.initialize_model()
            except ValueError as e:
                return Response({"error": str(e)}, status=500)

        # 입력 JSON 파일 로드
        try:
            with open(input_json_path, "r", encoding="utf-8") as input_file:
                data = json.load(input_file)
            logger.info(f"Input JSON loaded successfully with {len(data)} entries.")
        except Exception as e:
            logger.error(f"Failed to load input JSON: {str(e)}")
            return Response({"error": f"Failed to load input JSON: {str(e)}"}, status=500)

        processed_data = []

        # Llama 모델을 사용하여 각 항목 평가
        for entry in data:
            instruction = entry.get("instruction", "").strip()
            input_text = entry.get("input", "").strip()

            if not instruction or not input_text:
                logger.warning(f"Skipping invalid entry: {entry}")
                continue

            # 프롬프트 생성 (한글로 질문 작성)
            prompt = (
                f"Please analyze the following code and provide a comprehensive evaluation and suggestions for improvement:\n\n"
                f"File: {instruction}\n\n"
                f"Evaluation Criteria:\n"
                f"- Code readability: Is the code clear and easy to understand?\n"
                f"- Performance: Is the code optimized for speed and efficiency?\n"
                f"- Maintainability: Is the code modular and easy to modify or extend?\n"
                f"- Domain Identification: Identify the specific field or domain of the code (e.g., web development, machine learning, backend API, etc.) and explain your reasoning.\n\n"
                f"For each suggestion, include the following details:\n"
                f"- Specific areas for improvement: Highlight the exact parts of the code that need changes.\n"
                f"- Reasons for the changes: Provide a clear explanation of why the changes are necessary.\n"
                f"- Benefits of the changes: Explain how the suggested changes will enhance the code's quality or functionality.\n\n"
                f"Additionally, provide an overall evaluation score between 0 and 100 (0 = very poor, 100 = excellent):\n"
                f"- Assign a score based on the above criteria.\n"
                f"- Include a detailed explanation justifying the assigned score.\n\n"
                f"Code:\n{input_text}\n\n"
                f"Output:\n"
                f"- Domain of the Code:\n"
                f"- Suggestions for Improvement:\n"
                f"- Reasons for Changes:\n"
                f"- Benefits of Changes:\n"
                f"- Overall Quality Score (0-100) and Detailed Explanation:"
            )


            try:
                # 입력 데이터 토크나이징
                inputs = self.__class__.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=4096
                )

                # 모델 평가 수행
                outputs = self.__class__.model.generate(
                    **inputs, max_new_tokens=4096, num_return_sequences=1, no_repeat_ngram_size=2
                )
                generated_text = self.__class__.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                logger.info(f"Evaluation success for entry: {instruction}")
            except Exception as e:
                generated_text = f"Error during evaluation: {str(e)}"
                logger.error(f"Error during evaluation for entry {instruction}: {str(e)}")

            # 결과 데이터 저장
            processed_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": generated_text
            })

        # 평가 결과 JSON 파일로 저장
        try:
            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(processed_data, output_file, indent=4, ensure_ascii=False)
            logger.info(f"Results saved successfully to {output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save output JSON: {str(e)}")
            return Response({"error": f"Failed to save output JSON: {str(e)}"}, status=500)

        return Response({
            "message": "AI 평가가 성공적으로 완료되었습니다.",
            "output_file": output_json_path
        })
