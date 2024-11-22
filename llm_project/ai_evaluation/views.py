import json
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

class AIEvaluationAPIView(APIView):
    """
    Starcoder API를 사용해 코드를 평가하고 결과를 JSON 파일로 저장
    """
    @swagger_auto_schema(
        operation_description="AI 평가 작업 수행",
        responses={200: openapi.Response('성공 메시지와 JSON 파일 경로')}
    )
    def post(self, request):
        input_json_path = "./starcoder_input.json"
        output_json_path = "./llama3_input.json"
        api_url = "https://api-inference.huggingface.co/models/bigcode/starcoder2-3b"
        headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

        def query_starcoder_api(code, api_url, headers):
            payload = {"inputs": code}
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}, {response.text}")
                return None

        with open(input_json_path, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)
        
        processed_data = []
        for entry in data:
            prompt = f"""
            Please evaluate the following code:
            File Path: {entry['file_path']}
            Code: {entry['code']}
            Provide suggestions for improvement in JSON format.
            """
            result = query_starcoder_api(prompt, api_url, headers)
            if result:
                processed_data.append({
                    "instruction": f"Evaluate the code in {entry['file_path']}",
                    "input": entry["code"],
                    "output": result
                })
        
        with open(output_json_path, "w", encoding="utf-8") as output_file:
            json.dump(processed_data, output_file, indent=4, ensure_ascii=False)
        print(f"Processed data saved at {output_json_path}")

        return Response({"message": "AI evaluation completed successfully", "output_file": output_json_path})
