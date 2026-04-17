from setuptools import setup, find_packages

setup(
    name="stock_project",
    version="0.1.0",
    description="Stock price analysis, prediction, and risk assessment project",
    author="Jincheng Zhu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.40",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.9",
)