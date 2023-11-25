from setuptools import setup, find_packages

setup(
    name='pymeso',
    version='0.1',    
    description='This package provide researches in the area of mesoscale modelling for concrete, composite materials to easily create mesoscale models. You will need numpy, matplotlib, shapely',
    url='https://github.com/Faisallm/pymeso',
    author='Faisal Lawan Muhammad',
    author_email='Faisallawan1997@gmail.com',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)