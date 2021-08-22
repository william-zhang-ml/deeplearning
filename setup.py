from setuptools import setup


setup(
    name='wz-torch',
    version='0.0.1',
    url="https://github.com/william-zhang-ml/deeplearning",
    author="William Zhang",
    author_email="william.zhang.ml@gmail.com",
    packages=['wz_torch'],
    install_requires=[
        'torch>=1.8.1',
        'numpy>=1.18.5'
    ]
)
