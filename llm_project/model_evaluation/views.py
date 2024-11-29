import os
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from drf_yasg.utils import swagger_auto_schema
import logging
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AIEvaluationAPIView(APIView):
    """
    전처리된 데이터를 사용자 정의 모델에 입력하여 평가하는 API
    """
    model_dir = "./custom_llama3_model"  # 사용자 정의 모델 경로
    tokenizer = None
    model = None

    @classmethod
    def initialize_model(cls):
        """
        사용자 정의 모델과 토크나이저를 초기화하는 메서드
        """
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_dir)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_dir)
            logger.info("Custom model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"모델 로드 오류: {str(e)}")
            raise e

    @swagger_auto_schema(
        operation_description="사용자 정의 Llama3 모델을 사용하여 데이터를 평가합니다.",
        responses={200: openapi.Response("AI 평가 완료 메시지와 결과 JSON 경로")}
    )
    def post(self, request):
        """
        POST 요청으로 평가를 실행하고 결과를 JSON 파일로 저장
        """
        input_json_path = "./processed_input.json"
        output_json_path = "./evaluation_output.json"

        if not os.path.exists(input_json_path):
            return Response({"error": "전처리된 입력 파일이 존재하지 않습니다."}, status=400)

        # 모델 초기화
        if not self.__class__.tokenizer or not self.__class__.model:
            self.__class__.initialize_model()

        # 입력 파일 로드
        try:
            with open(input_json_path, "r", encoding="utf-8") as input_file:
                data = json.load(input_file)
            logger.info(f"Input JSON loaded successfully with {len(data)} entries.")
        except Exception as e:
            logger.error(f"입력 JSON 파일 로드 오류: {str(e)}")
            return Response({"error": f"입력 JSON 파일 로드 오류: {str(e)}"}, status=500)

        # 모델 평가
        processed_results = []
        for entry in data:
            prompt = f"{entry['instruction']}\n{entry['input']}"
            try:
                inputs = self.__class__.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                outputs = self.__class__.model.generate(**inputs, max_length=512)
                generated_text = self.__class__.tokenizer.decode(outputs[0], skip_special_tokens=True)
                entry["output"] = generated_text
                processed_results.append(entry)
            except Exception as e:
                logger.error(f"모델 평가 오류: {str(e)}")
                entry["output"] = f"Error during evaluation: {str(e)}"

        # 결과 저장
        try:
            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(processed_results, output_file, indent=4, ensure_ascii=False)
            logger.info(f"결과 저장 성공: {output_json_path}")
        except Exception as e:
            return Response({"error": f"결과 JSON 저장 오류: {str(e)}"}, status=500)

        return Response({
            "message": "AI 평가가 성공적으로 완료되었습니다.",
            "output_file": output_json_path
        })
