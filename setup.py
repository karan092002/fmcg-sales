from setuptools import setup, find_packages

setup(
    name="fmcg-profit-predictor",
    version="0.1.0",
    description="ML pipeline for predicting order-level profit across FMCG sales data",
    author="",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "streamlit>=1.28",
        "jupyter>=1.0",
    ],
    python_requires=">=3.10",
)
