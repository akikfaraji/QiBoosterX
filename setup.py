from setuptools import setup, find_packages

# Read long_description from README.md
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Genuine training boosters for low-end devices."

setup(
    name='qiboosterx',
    version='0.2.0',
    author='Akik Forazi',
    author_email='akikforaziinchaos@gmail.com',
    description='Genuine training boosters for low-end devices: Lookahead, SAM, SGLD, SWA, mixed precision, gradient accumulation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Akik-Forazi/QiBoosterX',
    project_urls={
        'Documentation': 'https://github.com/Akik-Forazi/QiBoosterX#readme',
        'Source': 'https://github.com/Akik-Forazi/QiBoosterX',
        'Bug Tracker': 'https://github.com/Akik-Forazi/QiBoosterX/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='lookahead sam sgld swa mixed-precision gradient-accumulation training-boosters pytorch',
    packages=find_packages(exclude=['tests', 'benchmarks', 'examples']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10',
        'numpy',
        'tqdm',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'torchvision'],
        'examples': ['torchvision'],
    },
    entry_points={
        'console_scripts': [
            'qiboostx=qiboosterx.cli:main',
        ],
    },
    include_package_data=True,
)
