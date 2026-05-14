from setuptools import setup, find_packages

setup(
    name="antigravity-iot-anomaly",
    version="1.0.0",
    description="End-to-End IoT Anomaly Detection System with Bi-LSTM Autoencoder and XGBoost",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "joblib>=1.3.0",
        "tensorflow>=2.13.0",
        "xgboost>=1.7.0",
        "kafka-python>=2.0.2",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "httpx>=0.25.0"],
    },
)
