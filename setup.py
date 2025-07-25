from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

try:
    long_description=open("README.md", encoding="utf-8").read()
except FileNotFoundError:
    long_description = ""

setup(
    name="imgcapgen",
    version="0.1.0",
    author="Srinivasa Chandakhanna",
    author_email="srinivasa.chandakanna@gmail.com",
    description="A Python package for image caption generation using CNN+LSTM and transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srinivasa-chandakanna/image-captioning",
    packages=find_packages(include=["imgcapgen", "imgcapgen.*"]),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)