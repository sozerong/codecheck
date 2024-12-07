import os
import json
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AIEvaluationAPIView(APIView):
    """
    Llama3 모델을 사용하여 전처리된 데이터를 평가하고 개선된 코드를 생성하는 API
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
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, token=api_key)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, token=api_key)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise e

    @swagger_auto_schema(
        operation_description="전처리된 데이터로 Llama3 모델에서 개선된 코드를 생성하고 결과를 저장합니다.",
        responses={200: openapi.Response("AI 평가 완료 메시지와 결과 JSON 경로")}
    )
    def post(self, request):
        """
        POST 요청으로 Llama3 모델 평가 실행 및 개선된 코드 저장
        """
        input_json_path = "./llama_input.jsonl"
        output_json_path = "./llama_output.jsonl"

        # 모델 초기화 확인
        if not self.__class__.tokenizer or not self.__class__.model:
            try:
                self.__class__.initialize_model()
            except Exception as e:
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

        # 각 항목 평가 및 개선된 코드 생성
        for entry in data:
            original_code = entry.get("original_code", "").strip()
            description = entry.get("description", "Analyze and improve the code.").strip()

            if not original_code:
                logger.warning(f"Skipping invalid entry: {entry}")
                continue

            # Llama3 평가용 프롬프트 생성
            prompt = (
                f"### Original Code:\n{original_code}\n\n"
                f"### Task:\n{description}\n\n"
                f"### Response Format:\n"
                f"- Suggestions: Provide specific recommendations to improve the code.\n"
                f"- Reasons: Explain why these changes are beneficial.\n"
                f"- Improved Code: Provide an improved version of the code.\n"
                f"- Score: Assign an overall score (0 to 100) for the original code.\n\n"
                f"### Response:\n"
                f"- Suggestions:\n- Reasons:\n- Improved Code:\n- Score:"
            )

            try:
                # 모델 입력 생성
                inputs = self.__class__.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )

                # Llama3 모델 출력 생성
                outputs = self.__class__.model.generate(
                    **inputs, max_new_tokens=512, num_return_sequences=1, no_repeat_ngram_size=2
                )
                generated_text = self.__class__.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                logger.info(f"Generated response for entry.")
            except Exception as e:
                generated_text = f"Error during evaluation: {str(e)}"
                logger.error(f"Error during evaluation for entry: {str(e)}")

            # 평가 결과 저장
            processed_data.append({
                "original_code": original_code,
                "description": description,
                "response": generated_text
            })

        # 결과 JSON 파일로 저장
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
