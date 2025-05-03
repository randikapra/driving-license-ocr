import os

project_name = "driving-license-ocr"

structure = {
    "data/raw": [],
    "data/processed": [],
    "data/samples": [],
    "notebooks": ["01_data_exploration.ipynb", "02_preprocessing.ipynb", "03_ocr_extraction.ipynb"],
    "src/data": ["loader.py", "preprocess.py", "__init__.py"],
    "src/ocr": ["ocr_engine.py", "postprocess.py", "__init__.py"],
    "src/parser": ["date_parser.py", "class_parser.py", "__init__.py"],
    "src/utils": ["helpers.py", "__init__.py"],
    "src": ["pipeline.py"],
    "tests": ["test_preprocessing.py", "test_ocr_engine.py", "test_parser.py"],
    "outputs/tables": [],
    "outputs/logs": [],
    "models/tesseract_custom": [],
    "scripts": ["run_pipeline.py", "preprocess_images.py", "extract_text.py", "extract_fields.py"],
    "docs": ["architecture.md"]
}

base_files = {
    "README.md": "# Driving License OCR Pipeline\n\nDescription here...",
    ".gitignore": "__pycache__/\n*.pyc\n.env\noutputs/\n",
    "requirements.txt": "opencv-python\npytesseract\npandas\nnumpy\nscikit-image\n",
    "setup.py": """from setuptools import setup, find_packages

setup(
    name="driving_license_ocr",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python", "pytesseract", "pandas", "numpy", "scikit-image"
    ],
)
""",
    "LICENSE": "MIT License\n\n[Add your license text here]"
}

def create_structure(base_path, structure):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for file in files:
            open(os.path.join(folder_path, file), 'a').close()

def create_base_files(base_path, files):
    for filename, content in files.items():
        with open(os.path.join(base_path, filename), 'w') as f:
            f.write(content)

def main():
    os.makedirs(project_name, exist_ok=True)
    create_structure(project_name, structure)
    create_base_files(project_name, base_files)
    print(f"✅ Project '{project_name}' scaffolded successfully!")

if __name__ == "__main__":
    main()
