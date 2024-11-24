import os
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import zipfile
import tarfile

class DataPreprocessingAPIView(APIView):
    """
    데이터 전처리 API: 압축된 파일(.zip, .tar.gz)을 업로드받아 파일 구조를 분석하고 전처리합니다.
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
        responses={200: openapi.Response("Data preprocessing completed successfully.")}
    )
    def post(self, request):
        # 파일 업로드 처리
        uploaded_file = request.FILES.get("file")  # 클라이언트에서 업로드한 파일
        if not uploaded_file: 
            return Response({"error": "No file uploaded."}, status=400)

        # 압축 파일 저장
        upload_dir = "./uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # 압축 해제 디렉토리
        extract_dir = "./extracted_files"
        os.makedirs(extract_dir, exist_ok=True)

        # 압축 해제
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            return Response({"error": "Unsupported file format."}, status=400)

        # 파일 구조 탐색
        def explore_file_structure(root_dir):
            file_structure = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_structure.append(file_path)
            return file_structure

        # 학습에 필요한 파일 필터링
        def filter_files(file_list, extensions=[".py", ".jsx", ".js", ".java", ".html", ".css"]):
            return [file for file in file_list if os.path.splitext(file)[1] in extensions]

        all_files = explore_file_structure(extract_dir)
        filtered_files = filter_files(all_files)

        # JSON 데이터 생성
        output_json_path = "./starcoder_input.json"
        data = []
        for file_path in filtered_files:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    code_content = file.read()
                    data.append({
                        "file_path": file_path,
                        "code": code_content
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        return Response({
            "message": "Data preprocessing completed successfully.",
            "output_file": output_json_path,
            "processed_files": len(filtered_files)
        })