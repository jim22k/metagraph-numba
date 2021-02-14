import pytest
import metagraph as mg
from metagraph import abstract_algorithm, concrete_algorithm, PluginRegistry
from metagraph.core.resolver import Resolver
from metagraph.core.dask.resolver import DaskResolver
import numpy as np
from metagraph.tests.util import default_plugin_resolver
from metagraph_numba.compiler import NumbaCompiler, SymbolTable, construct_call_wrapper_text, compile_wrapper


def test_register(default_plugin_resolver):
    assert "numba" in default_plugin_resolver.compilers


def test_compile_algorithm(res):
    a = np.arange(100)
    ret = res.algos.testing.scale(a, 4.0)
    np.testing.assert_array_equal(ret, a * 4.0)


def test_symbol_table_counters():
    tbl = SymbolTable()

    assert tbl.next_var() == "var0"
    assert tbl.next_const() == "const0"
    assert tbl.next_var() == "var1"
    assert tbl.next_const() == "const1"
    assert tbl.next_func() == "func0"
    assert tbl.next_func() == "func1"
    assert tbl.next_ret() == "ret0"
    assert tbl.next_ret() == "ret1"


def test_symbol_table_register_var():
    tbl = SymbolTable()

    tbl.register_var("vkey0")
    tbl.register_var("vkey1")

    assert tbl.var_key_to_sym == dict(vkey0="var0", vkey1="var1")
    assert tbl.var_sym_to_key == dict(var0="vkey0", var1="vkey1")


def test_symbol_table_find_symbol():
    tbl = SymbolTable()
    tbl.register_var("vkey0")
    algo0 = lambda x: x
    tbl.register_func("algo0", algo0, ["vkey0"])

    assert tbl.find_symbol("algo0") == "ret0"
    assert tbl.find_symbol("vkey0") == "var0"
    assert tbl.find_symbol("not_there") is None
    assert tbl.find_symbol([4]) is None


def test_symbol_table_register_func(ex1):
    tbl, (algo0, algo1) = ex1

    assert tbl.var_key_to_sym == dict(input0="var0", input1="var1")
    assert tbl.var_sym_to_key == dict(var0="input0", var1="input1")
    assert tbl.const_sym_to_value == dict(const0=2, const1=5)
    assert tbl.func_sym_to_key == dict(func0="algo0", func1="algo1")
    assert tbl.func_key_to_sym == dict(algo0="func0", algo1="func1")
    assert tbl.func_sym_to_func == dict(func0=algo0, func1=algo1)
    assert tbl.func_sym_to_ret_sym == dict(func0="ret0", func1="ret1")
    assert tbl.func_sym_to_args_sym == dict(
        func0=("var0", "var1", "const0"), func1=("ret0", "const1")
    )

    return tbl


def test_construct_call_wrapper_text(ex1):
    tbl, (algo0, algo1) = ex1
    text, wrapper_globals = construct_call_wrapper_text(
        wrapper_name="subgraph0",
        symbol_table=tbl,
        input_keys=["input0", "input1"],
        execute_keys=["algo0", "algo1"],
        output_key="algo1",
    )

    expected_text = '''\
def subgraph0(var0, var1):
    global const0
    global const1
    global func0
    global func1

    ret0 = func0(var0, var1, const0)
    ret1 = func1(ret0, const1)

    return ret1
'''
    expected_globals = {
        'const0': 2,
        'const1': 5,
        'func0': algo0,
        'func1': algo1,
    }
    assert expected_text == text
    assert expected_globals == wrapper_globals


def test_compile_wrapper(ex1):
    tbl, (algo0, algo1) = ex1
    text, wrapper_globals = construct_call_wrapper_text(
        wrapper_name="subgraph0",
        symbol_table=tbl,
        input_keys=["input0", "input1"],
        execute_keys=["algo0", "algo1"],
        output_key="algo1",
    )

    wrapper = compile_wrapper('subgraph0', text, wrapper_globals)
    for i0, i1 in [(10, 6), (12, 18), (-5, 2)]:
        assert wrapper(i0, i1) == (((i0 - i1) * 2) + 5)

def test_compile_subgraph(dres):
    a = np.arange(100)
    scale_func = dres.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)
    ans = dres.algos.testing.add(y, z)

    expected = (a * 2.0 * 3.0 * 4.0) + (a * 2.0 * 3.0)
    ret = ans.compute()
    np.testing.assert_array_equal(ret, expected)


@pytest.fixture
def ex1():
    tbl = SymbolTable()
    tbl.register_var("input0")
    tbl.register_var("input1")

    algo0 = lambda x, y, z: (x - y) * z
    algo1 = lambda x, y: x + y
    tbl.register_func("algo0", algo0, ["input0", "input1", 2])
    tbl.register_func("algo1", algo1, ["algo0", 5])

    return tbl, (algo0, algo1)


@pytest.fixture
def res():
    from metagraph.plugins.core.types import Vector
    from metagraph.plugins.numpy.types import NumpyVectorType

    @abstract_algorithm("testing.add")
    def testing_add(a: Vector, b: Vector) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add", compiler="numba")
    def compiled_add(a: NumpyVectorType, b: NumpyVectorType) -> NumpyVectorType:  # pragma: no cover
        return a + b

    @abstract_algorithm("testing.scale")
    def testing_scale(a: Vector, scale: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.scale", compiler="numba")
    def compiled_scale(a: NumpyVectorType, scale: float) -> NumpyVectorType:  # pragma: no cover
        return a * scale

    @abstract_algorithm("testing.offset")
    def testing_offset(a: Vector, *, offset: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.offset", compiler="identity_comp")
    def compiled_offset(a: NumpyVectorType, *, offset: float) -> NumpyVectorType:  # pragma: no cover
        return a + offset

    registry = PluginRegistry("test_subgraphs_plugin")
    registry.register(testing_add)
    registry.register(compiled_add)
    registry.register(testing_scale)
    registry.register(compiled_scale)
    registry.register(testing_offset)
    registry.register(compiled_offset)

    resolver = Resolver()
    # NumbaCompiler will be picked up from environment
    resolver.load_plugins_from_environment()
    resolver.register(registry.plugins)

    return resolver


@pytest.fixture
def dres(res):
    return DaskResolver(res)
