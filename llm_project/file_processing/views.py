import os
import json
import zipfile
import tarfile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

class FileProcessingAPIView(APIView):
    """
    평가용 압축파일을 해체하고 전처리하는 API
    """
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        operation_description="압축된 파일(.zip, .tar.gz)을 업로드하여 파일 구조를 분석하고 전처리합니다.",
        manual_parameters=[
            openapi.Parameter(
                "file",
                openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                description="압축된 파일(.zip 또는 .tar.gz)"
            )
        ],
        responses={200: openapi.Response("파일 전처리 완료 메시지와 결과 JSON 경로")}
    )
    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "파일이 업로드되지 않았습니다."}, status=400)

        # 디렉토리 설정
        upload_dir = "./uploaded_files"
        extract_dir = "./extracted_files"
        output_json_path = "./processed_input.json"
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)

        # 업로드된 파일 저장
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # 압축 해체
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif file_path.endswith(".tar.gz"):
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                return Response({"error": "지원하지 않는 파일 형식입니다."}, status=400)
        except Exception as e:
            return Response({"error": f"파일 해체 중 오류 발생: {str(e)}"}, status=500)

        # 전처리: 파일 구조를 분석하고 JSON 생성
        processed_data = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code_content = f.read()
                        processed_data.append({
                            "instruction": f"Analyze the code in {file}",
                            "input": code_content,
                            "output": ""
                        })
                except Exception as e:
                    print(f"파일 읽기 오류: {file_path}, {e}")

        try:
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                json.dump(processed_data, json_file, indent=4, ensure_ascii=False)
        except Exception as e:
            return Response({"error": f"JSON 저장 오류: {str(e)}"}, status=500)

        return Response({
            "message": "파일 전처리가 성공적으로 완료되었습니다.",
            "output_file": output_json_path
        })
