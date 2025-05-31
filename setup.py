from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hems",
    version="1.0.0",
    author="Rajarshi",
    author_email="santu1901@kgpian.iitkgp.ac.in",
    description="Home Energy Management System using SNN and RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rajarshi012003/SynaptiGridAI",
    packages=find_packages(include=["hems", "hems.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.1.0",
        "gym>=0.26.0",
        "cvxpy>=1.3.0",
        "streamlit>=1.24.0",
        "scikit-learn>=1.2.0",
        "snntorch>=0.7.0",
    ],
) 