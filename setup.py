#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages
from setuptools import setup

NAME =               'drosobot'
VERSION =            '0.1.0'
AUTHOR =             'M. K. Turkcan'
AUTHOR_EMAIL =       ''
URL =                ''
MAINTAINER =         ''
MAINTAINER_EMAIL =   ''
DESCRIPTION =        ''
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'MIT'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Database :: Front-Ends',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']
NAMESPACE_PACKAGES = ['drosobot']
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        namespace_packages = NAMESPACE_PACKAGES,
        packages = PACKAGES,
        include_package_data = True,
        install_requires = [
            'networkx>=2.4',
            'farm-haystack>=1.4.0'],
    )
