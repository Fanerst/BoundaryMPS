from setuptools import setup

setup(
    name='BoundayMPS',
    version='0.1.0',
    author='Feng Pan',
    author_email='fan_physics@126.com',
    packages=['BoundaryMPS'],# , 'artensor.tests'],
    # scripts=['bin/script1','bin/script2'],
    url='https://github.com/Fanerst/BoundaryMPS',
    license='LICENSE',
    description='Boundary Matrix Product State algorithm and its variants for solving graphical models defined on 2D lattice',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "torch",
    ],
)