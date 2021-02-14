import numba
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable
from dataclasses import dataclass, field


@dataclass
class SymbolTable:
    var_sym_to_key: Dict[str, Hashable] = field(default_factory=dict)
    var_key_to_sym: Dict[Hashable, str] = field(default_factory=dict)
    const_sym_to_value: Dict[str, Any] = field(default_factory=dict)
    func_sym_to_key: Dict[str, Any] = field(default_factory=dict)
    func_key_to_sym: Dict[Hashable, str] = field(default_factory=dict)
    func_sym_to_func: Dict[str, Callable] = field(default_factory=dict)
    func_sym_to_ret_sym: Dict[str, str] = field(default_factory=dict)
    func_sym_to_args_sym: Dict[str, Tuple[str]] = field(default_factory=dict)

    var_counter: int = 0
    const_counter: int = 0
    func_counter: int = 0
    ret_counter: int = 0

    def next_var(self):
        return self._next_symbol("var")

    def next_const(self):
        return self._next_symbol("const")

    def next_func(self):
        return self._next_symbol("func")

    def next_ret(self):
        return self._next_symbol("ret")

    def _next_symbol(self, prefix):
        counter = prefix + "_counter"
        value = getattr(self, counter)
        setattr(self, counter, value + 1)
        return f"{prefix}{value}"

    def register_var(self, key):
        var_sym = self.next_var()
        self.var_sym_to_key[var_sym] = key
        self.var_key_to_sym[key] = var_sym
        return var_sym

    def register_const(self, value):
        const_sym = self.next_const()
        self.const_sym_to_value[const_sym] = value
        return const_sym

    def find_symbol(self, arg):
        if ishashable(arg):
            # is this a Dask key that maps to an input variable?
            sym = self.var_key_to_sym.get(arg, None)
            if sym is not None:
                return sym

            # is this a Dask key that maps to a function?
            func_sym = self.func_key_to_sym.get(arg, None)
            if func_sym is not None:
                # the symbol is the return value from calling the function
                return self.func_sym_to_ret_sym[func_sym]

        # return None means no symbol found

    def register_func(self, key, func, args):
        """'Must register functions in toposort order"""
        func_sym = self.next_func()
        ret_sym = self.next_ret()
        self.func_sym_to_key[func_sym] = key
        self.func_key_to_sym[key] = func_sym
        self.func_sym_to_func[func_sym] = func
        self.func_sym_to_ret_sym[func_sym] = ret_sym

        arg_sym_list = []
        for arg in args:
            arg_sym = self.find_symbol(arg)
            if arg_sym is None:
                arg_sym = self.register_const(arg)
            arg_sym_list.append(arg_sym)

        self.func_sym_to_args_sym[func_sym] = tuple(arg_sym_list)

        return func_sym, ret_sym


def construct_call_wrapper_text(
    wrapper_name: str,
    symbol_table: SymbolTable,
    input_keys: List[Hashable],
    execute_keys: List[Hashable],
    output_key: Hashable,
) -> Tuple[str, Dict[str, Any]]:

    wrapper_globals = symbol_table.const_sym_to_value.copy()
    wrapper_globals.update(symbol_table.func_sym_to_func)

    func_text = ""

    # call signature
    func_text += (
        f"def {wrapper_name}("
        + ", ".join([symbol_table.var_key_to_sym[ikey] for ikey in input_keys])
        + "):\n"
    )

    # declare globals
    for global_name in wrapper_globals:
        func_text += f"    global {global_name}\n"
    func_text += "\n"

    # body
    for ekey in execute_keys:
        func_sym = symbol_table.func_key_to_sym[ekey]
        ret_sym = symbol_table.func_sym_to_ret_sym[func_sym]
        args_sym = symbol_table.func_sym_to_args_sym[func_sym]
        func_text += f"    {ret_sym} = {func_sym}(" + ", ".join(args_sym) + ")\n"
    func_text += "\n"

    # return value
    func_sym = symbol_table.func_key_to_sym[output_key]
    ret_sym = symbol_table.func_sym_to_ret_sym[func_sym]
    func_text += f"    return {ret_sym}\n"

    return func_text, wrapper_globals


def compile_wrapper(
    wrapper_name: str, wrapper_text: str, wrapper_globals: Dict[str, Any]
):
    my_globals = wrapper_globals.copy()
    exec(wrapper_text, my_globals)
    return my_globals[wrapper_name]


class NumbaCompiler(Compiler):
    """This compiler compiles functions using Numba.
    """

    def __init__(self, name="numba"):
        super().__init__(name=name)
        self._subgraph_count = 0

    def compile_algorithm(self, algo, literals):
        return numba.jit(algo.func)

    def compile_subgraph(self, subgraph: Dict, inputs: List[str], output: str):
        tbl = SymbolTable()

        # register the inputs as variables
        for key in inputs:
            tbl.register_var(key)

        # register each function in the subgraph
        for key, task in subgraph.items():
            # all metagraph tasks are in (func, args, kwargs) format
            delayed_algo, args, kwargs = task
            if len(kwargs) != 0:
                raise CompileError(
                    "NumbaCompiler only supports functions bound kwargs.\n"
                    f"When compiling:\n{delayed_algo.func_label}\nfound kwargs:\n{kwargs}"
                )
            jit_func = numba.jit(inline="always")(delayed_algo.algo.func)
            tbl.register_func(key, jit_func, args)

        toposort_keys = list(toposort(subgraph))
        subgraph_wrapper_name = "subgraph" + str(self._subgraph_count)
        self._subgraph_count += 1
        wrapper_text, wrapper_globals = construct_call_wrapper_text(
            name=subgraph_wrapper_name,
            symbol_table=tbl,
            input_keys=inputs,
            execute_keys=toposort_keys,
            output_key=output,
        )

        wrapper_func = compile_wrapper(
            subgraph_wrapper_name, wrapper_text, wrapper_globals
        )
        return wrapper_func
