import json
import requests
import os
from dotenv import load_dotenv  # 환경 변수 로드
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# .env 파일 로드
load_dotenv()

class AIEvaluationAPIView(APIView):
    """
    Starcoder API를 사용해 코드를 평가하고 결과를 JSON 파일로 저장
    """

    @swagger_auto_schema(
        operation_description="AI 평가 작업 수행. Starcoder API를 호출하여 코드를 평가하고 결과를 JSON 파일로 저장합니다.",
        responses={200: openapi.Response("성공 메시지와 JSON 파일 경로")}
    )
    def post(self, request):
        input_json_path = "./starcoder_input.json"
        output_json_path = "./llama3_input.json"
        api_url = "https://api-inference.huggingface.co/models/bigcode/starcoder2-3b"

        # Hugging Face API 키 로드
        api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if not api_key:
            return Response(
                {"error": "Hugging Face API key is missing. Please set it in the .env file."},
                status=500
            )

        headers = {"Authorization": f"Bearer {api_key}"}

        def query_starcoder_api(code_snippet, api_url, headers):
            """
            Starcoder API에 요청을 보내고 응답을 반환
            """
            payload = {"inputs": code_snippet}
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                print(f"Request Payload: {payload}")
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content: {response.text}")

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Error Details: {response.text}")
                    return {
                        "error": f"API call failed with status {response.status_code}",
                        "details": response.text
                    }
            except Exception as e:
                print(f"Exception during API call: {str(e)}")
                return {"error": str(e)}

        def preprocess_code_snippet(code_snippet, max_length=1000):
            """
            코드 스니펫을 전처리하여 API 입력값에 맞게 제한
            """
            if not code_snippet.strip():
                return None  # 비어 있는 코드 스니펫 무시
            return code_snippet[:max_length]

        # 입력 데이터 로드
        try:
            with open(input_json_path, "r", encoding="utf-8") as input_file:
                data = json.load(input_file)
        except FileNotFoundError:
            return Response(
                {"error": f"Input JSON file '{input_json_path}' not found."},
                status=404
            )
        except json.JSONDecodeError as e:
            return Response(
                {"error": f"Failed to parse JSON file: {str(e)}"},
                status=400
            )

        processed_data = []

        # Starcoder API 호출 및 처리
        for entry in data:
            # 입력 코드 검증 및 전처리
            code_snippet = preprocess_code_snippet(entry["code"])
            if not code_snippet:
                print(f"Skipping empty or invalid code block in {entry['file_path']}")
                continue

            # 간단한 프롬프트 생성
            prompt = f"""
            ### Analyze the following code:
            {code_snippet}
            ### Provide suggestions for improvement:
            """
            print(f"Generated Prompt: {prompt}")

            # Starcoder API 호출
            result = query_starcoder_api(prompt, api_url, headers)

            # API 응답 처리
            if "generated_text" in result:
                suggestions = result["generated_text"]
            else:
                suggestions = result.get("error", "Failed to process this code block.")

            processed_data.append({
                "instruction": f"Evaluate the code in {entry['file_path']}",
                "input": entry["code"],
                "output": suggestions
            })

        # 결과 JSON 저장
        try:
            with open(output_json_path, "w", encoding="utf-8") as output_file:
                json.dump(processed_data, output_file, indent=4, ensure_ascii=False)
                print(f"Processed data saved at {output_json_path}")
        except Exception as e:
            return Response(
                {"error": f"Failed to save output JSON: {str(e)}"},
                status=500
            )

        return Response({"message": "AI evaluation completed successfully", "output_file": output_json_path})
