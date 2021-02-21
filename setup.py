from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph-numba",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="python-numba plugins for Metagraph",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph_numba", "metagraph_numba.*"]),
    include_package_data=True,
    install_requires=["metagraph", "numba"],
    entry_points={"metagraph.plugins": "plugins=metagraph_numba.registry:find_plugins"},
)
