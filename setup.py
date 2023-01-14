
from setuptools import setup, find_packages

_version_dict = dict()
with open('loox/version.py') as f:
    exec(f.read(), _version_dict)
__version__ = _version_dict['__version__']

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires = [
        'jax>=0.1',
        'einops>=0.3'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ]
)

