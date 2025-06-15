from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return [req for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name='contextkernel',
    version='0.1.0', # Initial development version
    packages=find_packages(include=['contextkernel', 'contextkernel.*']),
    description='A modular AI system that thinks like a brain — it remembers, understands, and routes information intelligently.',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name / Organization', # Replace with actual author
    author_email='your.email@example.com', # Replace with actual email
    url='https://github.com/your-repo/contextkernel', # Replace with actual repo URL
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.9',
)
