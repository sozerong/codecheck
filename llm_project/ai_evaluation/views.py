import os
import json
import requests
import boto3
from botocore.exceptions import ClientError
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging

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
        try:
            # AWS Parameter Store에서 Hugging Face API 키 가져오기
            api_key = get_aws_parameter("CodeCheck")
            if not api_key:
                raise ValueError("Hugging Face API key is missing.")

            # 모델 및 토크나이저 로드
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_auth_token=api_key)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, use_auth_token=api_key)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    @swagger_auto_schema(
        operation_description="Llama3 모델을 사용하여 코드를 평가하고 JSON 파일로 저장하며 학습 API로 전송합니다.",
        responses={200: openapi.Response("AI 평가 완료 메시지와 결과 JSON 경로")}
    )
    def post(self, request):
        """
        POST 요청으로 평가를 실행하고 결과를 JSON 파일로 저장한 뒤 Llama 학습 API로 전송
        """
        input_json_path = "./llama3_input.json"
        output_json_path = "./llama3_output.json"
        llama_training_api_url = "http://localhost:8000/llama_training/train"  # Llama 학습 API URL

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

            # 프롬프트 생성 (세밀한 항목 포함)
            prompt = (
                f"Analyze the following code and provide detailed feedback in the following format:\n\n"
                f"Code:\n{input_text}\n\n"
                f"Output Format:\n"
                f"- Domain: Specify the field or domain of the code (e.g., web development, machine learning, backend API, etc.)\n"
                f"- Evaluation: Provide a detailed evaluation of the code, including readability, performance, and maintainability.\n"
                f"- Suggestions: Provide specific recommendations for improvement.\n"
                f"- Reasons: Explain why the changes are necessary and the expected benefits.\n"
                f"- Improved Code: Provide a rewritten version of the code incorporating the suggested improvements.\n"
                f"- Score: Assign an overall score between 0 and 100 and justify the score.\n\n"
                f"Code Analysis Output:\n"
                f"- Domain:\n"
                f"- Evaluation:\n"
                f"- Suggestions:\n"
                f"- Reasons:\n"
                f"- Improved Code:\n"
                f"- Score:"
            )

            try:
                # 입력 데이터 토크나이징
                inputs = self.__class__.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=4000
                )

                # 모델 평가 수행
                outputs = self.__class__.model.generate(
                    **inputs, max_new_tokens=4000, num_return_sequences=1, no_repeat_ngram_size=2
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

        # 결과를 Llama 학습 API로 전송
        try:
            with open(output_json_path, "rb") as json_file:
                response = requests.post(llama_training_api_url, files={"file": json_file})
            if response.status_code == 200:
                logger.info(f"Successfully sent results to Llama training API: {llama_training_api_url}")
            else:
                logger.error(f"Failed to send results to Llama training API: {response.status_code} - {response.text}")
                return Response({"error": "Failed to send results to Llama training API"}, status=500)
        except Exception as e:
            logger.error(f"Error sending results to Llama training API: {str(e)}")
            return Response({"error": f"Error sending results to Llama training API: {str(e)}"}, status=500)

        return Response({
            "message": "AI 평가가 성공적으로 완료되었고 Llama 학습 API로 전송되었습니다.",
            "output_file": output_json_path
        })
