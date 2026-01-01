from setuptools import setup, find_packages

setup(
    name='ade-sidechain',
    version='0.1.0',
    description='Python SDK for Ade Sidechain - AI Agent Integration',
    author='Ade Team',
    packages=find_packages(),
    install_requires=[
        'requests>=2.31.0',
        'ed25519>=1.5',
        'base58>=2.1.1',
        'pynacl>=1.5.0',
        'httpx>=0.25.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)


