import os

import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''

try:
    with open("requirements-dev.txt", "r") as fh:
        tests_require = [line for line in fh.read().split(os.linesep) if line]
except FileNotFoundError:
    tests_require = []

try:
    with open("requirements.txt", "r") as fh:
        install_requires = [line for line in fh.read().split(os.linesep) if line and not line.startswith('git')]
except FileNotFoundError:
    install_requires = []

setuptools.setup(
    name="cognitivexr-cpop",
    version="0.0.1.dev1",
    author="Christian Stippel, Thomas Rausch, Matthias Hürbe",
    author_email="christian.stippel@cognitivexr.at, thomas@cognitivexr.at, matthiashuerbe@gmail.com",
    description="CPOP: Cyber-Physical Object Positioning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cognitivexr/cpop",
    packages=setuptools.find_packages(),
    test_suite="tests",
    tests_require=tests_require,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
