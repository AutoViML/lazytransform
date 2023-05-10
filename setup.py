import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazytransform",
    version="1.1",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Clean your data using a scikit-learn transformer in a single line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/pandas_dq",
    py_modules = ["pandas_dq"],
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
    "numpy~=1.20.3",
    "pandas~=1.2.4",
    "scikit-learn~=0.24.2",
    "scipy~=1.6.3",
    "python-dateutil~=2.8.1",
    "lightgbm>=3.2.1",
    "imbalanced-learn>=0.8.0",
    "matplotlib~=3.4.2",
    "imbalanced-ensemble~=0.2.0",
    ],
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
