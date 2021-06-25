from setuptools import setup

# read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


# read dependency requirements
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()


setup(
    name='pyfew',
    version='0.0.1',
    license='MIT',
    author='wearables@ox',
    packages=['pyfew'],
    install_requires=[
        'catch22',
        'numpy',
        'scipy'
    ],
)

