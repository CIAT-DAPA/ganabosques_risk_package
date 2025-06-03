from setuptools import setup, find_packages

setup(
    name="ganabosques_risk_package",
    version="0.1.0",
    description="Package to estimate risk level related to deforestation and protected areas using MRV protocol from Colombia",
    author="Steven Sotelo",
    author_email="h.sotelo@cgiar.org",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.3",
        "geopandas==1.0.1",
        "shapely==2.1.1",
        "rasterio==1.4.3",
        "numpy==2.2.6",
        "tqdm==4.67.1",
        "fiona==1.10.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
