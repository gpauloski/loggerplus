import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="loggerplus",
    version="0.0.1",
    author="Greg Pauloski",
    author_email="jgpauloski@uchicago.edu",
    description="Custom Logger for Machine Learning Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpauloski/loggerplus",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'tensorboard',
    ]
)

