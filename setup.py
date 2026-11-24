from setuptools import setup, find_packages

setup(
    name="ganabosques_risk_package",
    version="0.0.11",
    description="Package to estimate alert related to deforestation and protected areas using MRV protocol from Colombia",
    author="Steven Sotelo",
    author_email="h.sotelo@cgiar.org",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        # Dependencias directas del código. Rangos (no versiones exactas) para no
        # chocar con otros paquetes del proyecto que instale esta librería.
        # El entorno exacto probado en CI está en requirements.txt.
        "geopandas>=1.0,<2",
        "numpy>=1.26,<3",
        "pandas>=2.1,<3",
        "pyproj>=3.6",
        "rasterio>=1.3",
        "rasterstats>=0.19",  # solo para los módulos obsoletos plot_alert_direct*
        "shapely>=2.0",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
