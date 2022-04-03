import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazytransform",
    version="0.26",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically build data and model pipelines using scikit-learn in a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/lazytransform",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy>=1.21.5",
        "pandas==1.3.5",
        "matplotlib",
        "scikit-learn>=0.24.2",
        "imbalanced-learn>=0.7",
        "category-encoders>=2.4.0",
        "xlrd",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
