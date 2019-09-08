import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='pytorch-utilities',
    version='0.0.1',
    author='Bradley Ezard',
    author_email='bradley.ezard@postgrad.curtin.edu.au',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bradezard131/pytorch-utilities',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
