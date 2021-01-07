from metagraph import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry("metagraph_numba")


def find_plugins():
    # Ensure we import all items we want registered
    from . import compiler

    # registry.register_from_modules(compiler)
    return registry.plugins
