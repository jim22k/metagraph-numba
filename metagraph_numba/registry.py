from metagraph import PluginRegistry


def find_plugins():
    # Ensure we import all items we want registered
    from . import compiler

    registry = PluginRegistry("metagraph_numba")
    registry.register(compiler.NumbaCompiler())
    return registry.plugins
