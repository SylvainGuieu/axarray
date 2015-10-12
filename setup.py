from distutils.core import setup

setup(
    # Application name:
    name='axarray',

    # Version number (initial):
    version='0.1.0',

    # Application author details:
    author='Sylvain Guieu',
    author_email='sylvain.guieu@gmail.com',

    # Packages
    #packages=['numpy'],

    # Include additional files into the package
    #include_package_data=True,

    # Details
    url='https://github.com/SylvainGuieu/axarray/',

    #
    license='LICENSE.txt',
    description='numpy array with labeled axes',

    long_description=open('README.md').read(),

    # Dependent packages (distributions)
    install_requires=[
        'numpy',
    ],

    keywords = 'array numpy axes', 
    classifiers=[
    # see https://pypi.python.org/pypi?%3Aaction=list_classifiers
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',

    # Pick your license as you wish (should match 'license' above)
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    ]
)