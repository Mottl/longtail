import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    INSTALL_REQUIRES = [l.split('#')[0].strip() for l in fh if not l.strip().startswith('#')]

setuptools.setup(
    name="longtail",
    version="0.0.1",
    author="Dmitry Mottl",
    author_email="dmitry.mottl@gmail.com",
    license="MIT",
    description="Transforms RV from the given empirical distribution to the standard normal",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mottl/longtail",
    keywords="random-variable-transformations probability-distribution probability-density-function",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=INSTALL_REQUIRES,
)
