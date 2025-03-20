from setuptools import setup
from Cython.Build import cythonize

setup(
    name='agent',
    #ext_modules=cythonize("agent.pyx", compiler_directives={'cdivision': True, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False}),
    ext_modules=cythonize("agent.pyx"),
)

setup(
    name='network',
    #ext_modules=cythonize("network.pyx",compiler_directives={'cdivision': True, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False}),
    ext_modules=cythonize("network.pyx"),
)

setup(
    name='pandemic',
    #ext_modules=cythonize("pandemic.pyx",compiler_directives={'cdivision': True, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False}),
    ext_modules=cythonize("pandemic.pyx"),
)

setup(
    name='world',
    #ext_modules=cythonize("world.pyx",compiler_directives={'cdivision': True, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False}),
    ext_modules=cythonize("world.pyx"),
)

setup(
    name='main',
    #ext_modules=cythonize("main.pyx",compiler_directives={'cdivision': True, 'boundscheck': False, 'wraparound': False, 'initializedcheck': False}),
    ext_modules=cythonize("main.pyx"),
)



