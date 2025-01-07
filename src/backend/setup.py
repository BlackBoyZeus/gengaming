#!/usr/bin/env python3

from setuptools import setup, find_packages  # setuptools v65.0.0+

# Read requirements with error handling
try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    raise RuntimeError('Critical: requirements.txt not found. Cannot proceed with installation.')
except Exception as e:
    raise RuntimeError(f'Error reading requirements.txt: {str(e)}')

# Read long description with error handling
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    print('Warning: README.md not found. Using default description.')
    long_description = 'GameGen-X backend service for real-time game video generation and control on FreeBSD-based Orbis OS'
except Exception as e:
    print(f'Warning: Error reading README.md: {str(e)}. Using default description.')
    long_description = 'GameGen-X backend service for real-time game video generation and control on FreeBSD-based Orbis OS'

setup(
    # Package metadata
    name='gamegen-x-backend',
    version='0.1.0',
    description='GameGen-X backend service for real-time game video generation and control on FreeBSD-based Orbis OS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='GameGen-X Team',
    license='MIT',
    
    # Package configuration
    packages=find_packages(
        exclude=[
            'tests*',
            'docs*',
            'examples*',
            'build*',
            'dist*',
            '*.egg-info'
        ]
    ),
    
    # Python version requirement
    python_requires='>=3.9',
    
    # Dependencies
    install_requires=requirements,
    
    # Platform and environment
    platforms=['FreeBSD'],
    
    # Package classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: BSD :: FreeBSD',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video :: Display',
        'Topic :: Games/Entertainment',
        'Environment :: FreeBSD',
        'Framework :: FastAPI',
        'Framework :: Pytorch'
    ],
    
    # Entry points for CLI tools (if needed)
    entry_points={
        'console_scripts': [
            'gamegen-x=backend.cli:main',
        ],
    },
    
    # Additional package data
    include_package_data=True,
    zip_safe=False,
    
    # Project URLs
    project_urls={
        'Source': 'https://github.com/gamegen-x/backend',
        'Documentation': 'https://docs.gamegen-x.dev',
        'Bug Tracker': 'https://github.com/gamegen-x/backend/issues',
    },
    
    # FreeBSD-specific options
    options={
        'bdist_wheel': {
            'plat_name': 'freebsd_13_amd64',
            'universal': False,
        }
    }
)