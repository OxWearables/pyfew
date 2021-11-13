from setuptools import setup

# read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# read dependency requirements
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()


setup(
    name="pyfew",
    version="0.0.1",
    license="MIT",
    author="wearables@ox",
    author_email="angerhangy@gmail.com",
    packages=["pyfew"],
    include_package_data=True,
    install_requires=["pyyaml", "catch22", "numpy", "scipy"],
)
