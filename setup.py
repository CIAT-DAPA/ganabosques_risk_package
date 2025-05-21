from setuptools import setup, find_packages

setup(
    name="ganabosques_risk_package",
    version="0.1.0",
    description="MRV risk calculation methodology package for Ganabosques",
    author="Ciencia de Datos GPT",
    author_email="h.sotelo@cgiar.org",
    packages=find_packages(),
    install_requires=[
        "pandas", "geopandas", "shapely", "rasterio", "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)