import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='asteroid_det',
    version='1.0',
    author='Noppachanin Kongsathitporn',
    description='Train and test binary classification models for asteroid detection',
    packages=['ast_det'],
    package_dir={'ast_det': 'src/'},
    install_requires=[
        "torch==1.12.0",
        "torchmetrics==0.10.0",
        "scikit-learn==1.1.2",
        "pandas>=0.25.1",
        "psycopg2-binary==2.8.3",
        "astropy==5.1.1",
        "matplotlib>=3.1.1",
    ],
    license='GNU General Public License v3 (GPLv3)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
         "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6,<3.8',
)