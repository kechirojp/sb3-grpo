from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sb3-grpo",
    version="0.1.0",
    author="kechirojp",
    author_email="kechirojp@gmail.com",
    description="Group Relative Policy Optimization (GRPO) implementation for Stable Baselines3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kechirojp/sb3-grpo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "torch>=1.11.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    keywords="reinforcement-learning, stable-baselines3, ppo, grpo, machine-learning",
)
