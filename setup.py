from setuptools import setup, find_packages

setup(
    name="deepAttention",
    version="0.1.0",
    author="Mercel Vubangsi",
    author_email="vmercel@outlook.fr",
    description="A Python package for single-target multi-class classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vmercel/deepAttention",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2.0.0",
        "keras>=2.0.0"
    ],
)

