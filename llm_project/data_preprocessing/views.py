import os
import json
import zipfile
import tarfile
import ast
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


class DataPreprocessingAPIView(APIView):
    """
    Data Preprocessing API: Unpacks uploaded compressed files and converts them into a dataset format.
    """
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        operation_description="Upload a compressed file (.zip, .tar.gz) to analyze its structure and preprocess the data.",
        manual_parameters=[
            openapi.Parameter(
                "file",
                openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                description="Compressed file (.zip or .tar.gz)"
            )
        ],
        responses={200: openapi.Response("Data preprocessing completed successfully.")}
    )
    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=400)

        upload_dir = "./uploaded_files"
        extract_dir = "./extracted_files"
        output_json_path = "./llama3_input.json"

        # Create directories
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Extract the compressed file
        try:
            if file_path.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif file_path.endswith(".tar.gz"):
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                return Response({"error": "Unsupported file format."}, status=400)
        except Exception as e:
            return Response({"error": f"Failed to extract files: {str(e)}"}, status=500)

        # Explore file structure
        def explore_file_structure(root_dir):
            file_structure = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_structure.append(file_path)
            return file_structure

        # Filter files by extension
        def filter_files(file_list, extensions=[".py", ".jsx", ".js", ".html", ".css"]):
            return [file for file in file_list if os.path.splitext(file)[1] in extensions]

        # Analyze Python code
        def analyze_python_code(code_content):
            try:
                tree = ast.parse(code_content)
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                return {
                    "functions": functions,
                    "classes": classes,
                    "total_lines": len(code_content.split("\n"))
                }
            except Exception:
                return {"functions": [], "classes": [], "total_lines": len(code_content.split("\n"))}

        # Analyze files and construct dataset
        all_files = explore_file_structure(extract_dir)
        filtered_files = filter_files(all_files)
        data = []

        for file_path in filtered_files:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    code_content = file.read()
                    metadata = analyze_python_code(code_content)

                    # 프롬프트에 개선점 요청과 평가 기준 포함
                    data.append({
                        "instruction": (
                            f"Analyze the following code in {os.path.basename(file_path)}. "
                            "Provide suggestions for improvement, focusing on code readability, performance, "
                            "and maintainability. For each suggestion, explain why the change is necessary "
                            "and the benefits it brings to the code. Finally, provide an evaluation score "
                            "from 1 to 10, where 1 is poor and 10 is excellent, based on overall code quality."
                        ),
                        "input": code_content,
                        "metadata": metadata,
                        "output": ""  # Llama 모델 출력이 여기에 들어감
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")


        # Save the dataset
        try:
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
        except Exception as e:
            return Response({"error": f"Failed to save dataset: {str(e)}"}, status=500)

        return Response({
            "message": "Data preprocessing completed successfully.",
            "output_file": output_json_path,
            "processed_files": len(filtered_files)
        })
