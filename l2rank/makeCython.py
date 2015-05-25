#i build cython package for lambdamart gradient computation
import sys
sys.argv+=["build_ext","--inplace"]

import os
os.system("cython -a lmart_aux.pyx")
import numpy




from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup

config = Configuration("", "",None)
config.add_extension("lmart_aux",sources=["lmart_aux.c"],include_dirs=[numpy.get_include()])

setup(**config.todict())