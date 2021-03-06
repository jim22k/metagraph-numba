"""
Numba compiler plugin:

This compiler plugin assumes that concrete algorithm implementations are
"Numba-ready".  Specifically:

* The decorated Python function is compilable by Numba.
* The input and output data types can be boxed/unboxed by Numba automatically.

To fuse multiple tasks from the subgraph into a single function, all of the
individual task functions are first wrapped in the Numba @jit decorator with
inlining enabled. Then a wrapper function that calls each of the tasks in the
correct order is generated.  The required inlining of the task functions into
the wrapper should give the compiler the most opportunity for loop fusion and
code optimization.

Task function arguments are sorted into one of three categories using the
SymbolTable class:

* Inputs passed from tasks outside the subgraph ("var0", "var1", etc)
* Values known at compile time ("const0", "const1", etc)
* Results from other tasks in the subgraph ("ret0", "ret1", etc)

The tasks in the subgraph are topologically sorted based on their dependencies
and inspected to build the symbol table.  The wrapper function is then
generated as text (for now) since Python doesn't have an easier way to
programmatically generate a function.  Finally, the function is materialized
in the interpreter by using exec() and populating the globals with the named
constants and functions that the wrapper references.  Numba will capture these
global values when compiling the function.

Note that while the fused function is generated ahead of time, the actual
Numba compilation (Python bytecode->machine code) is not triggered until the
fused task is executed by Dask at some point in the future.  This is an
implementation convenience, as ahead-of-time compilation would require
generating a detailed type signature for the function, which is difficult and
error prone.
"""

import numba
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable
from dataclasses import dataclass, field


@dataclass
class SymbolTable:
    """Container for mapping functions (identified by a dict key), Python
    arguments, and input parameters (also identified by a dict key) to valid
    Python identifiers.

    This class is needed because Dask keys only need to be hashable, but are
    not necessarily valid Python identifiers.  Also, Dask tasks reference
    values and other tasks, which need to be mapped to identifiers as well.

    The naming convention for attributes is as follows:
      - `sym`: A symbol name that is a valid Python identifier.  This class
        uses names like `constN`, `varN`, `funcN`, and `retN`.
      - `key`: A key for a task in a Dask graph
      - `func`: The callable function associated with a task
      - `ret`: The return value from a function
      - `var`: An external, unknown value, typically coming from Dask tasks
        defined outside of the subgraph being considered.
      - `const`: A Python value known at compile time that can be treated as a
        constant.

    The mappings in this class typically populated by calling register_var()
    and register_func().  See docstrings below for more details.

    Value symbols (`ret`, `var`, `const`) can have a type optionally
    associated with them.

    Note that automatically generated symbol names are only guaranteed to be
    unique for this instance of the symbol table.  Other symbol tables will
    collide with this one.
    """

    var_sym_to_key: Dict[str, Hashable] = field(default_factory=dict)
    var_key_to_sym: Dict[Hashable, str] = field(default_factory=dict)
    const_sym_to_value: Dict[str, Any] = field(default_factory=dict)
    func_sym_to_key: Dict[str, Any] = field(default_factory=dict)
    func_key_to_sym: Dict[Hashable, str] = field(default_factory=dict)
    func_sym_to_func: Dict[str, Callable] = field(default_factory=dict)
    func_sym_to_ret_sym: Dict[str, str] = field(default_factory=dict)
    func_sym_to_args_sym: Dict[str, Tuple[str]] = field(default_factory=dict)
    sym_to_type: Dict[str, Any] = field(default_factory=dict)

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

    def register_var(self, key, *, type=None):
        """Register an external values associated with `key`.
        
        Returns uniquely generated symbol name.
        """
        var_sym = self.next_var()
        self.var_sym_to_key[var_sym] = key
        self.var_key_to_sym[key] = var_sym
        if type is not None:
            self.sym_to_type[var_sym] = type
        return var_sym

    def register_const(self, value, *, type=None):
        """Register a Python object as a compile time constant.

        Returns uniquely generated symbol name.
        """
        const_sym = self.next_const()
        self.const_sym_to_value[const_sym] = value
        if type is not None:
            self.sym_to_type[const_sym] = type
        return const_sym

    def find_symbol(self, arg):
        """Find the symbol associated with a given function argument.

        If the argument is a known key for either an external value
        or a previously registered function call, the appropriate symbol
        (either `varN` or `retN`) for the value will be returned.

        If this argument is not a known key, return None
        """
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

    def register_func(self, key, func, args, *, arg_types=None, ret_type=None):
        """Register a function call with return value associated with key and
        the given argument list.

        The function symbol and return value symbol will be generated and
        returned. Arguments will be scanned and logged as `var` or `ret`
        symbols (if they are known Dask keys), or the argument value will be
        stored and a newly generated const symbol recorded in the argument
        list.

        Note that function calls must be registered in topologically sorted
        dependency order, or the key discovery will not work.
        """
        func_sym = self.next_func()
        ret_sym = self.next_ret()
        self.func_sym_to_key[func_sym] = key
        self.func_key_to_sym[key] = func_sym
        self.func_sym_to_func[func_sym] = func
        self.func_sym_to_ret_sym[func_sym] = ret_sym
        if ret_type is not None:
            self.sym_to_type[ret_sym] = ret_type

        arg_sym_list = []
        for iarg, arg in enumerate(args):
            arg_sym = self.find_symbol(arg)
            if arg_sym is None:
                if arg_types is not None:
                    arg_type = arg_types[iarg]
                else:
                    arg_type = None
                arg_sym = self.register_const(arg, type=arg_type)
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
    """Create the Python source code of a wrapper function for a subgraph.

    The function will be generated with the name `wrapper_name and take as
    input arguments the values associated with `input_keys` in the order
    given.  Functions will be executed in the order provided by
    `execute_keys`, and the function will return the result associated with
    `output_key`.  All tasks and inputs need to be recorded in the provided
    `symbol_table` before calling this function.  

    Because constants and function definitions cannot be serialized into the
    wrapper source in general, the wrapper will also indicate all of those
    symbols are globals using the `global` statement.  When using exec() to 
    parse the wrapper source, the const and func values will need to be
    put into the globals dict passed to exec().

    The return value of this function will look something like:

        def subgraph0(var0, var1):
            global const0
            global const1
            global func0
            global func1

            ret0 = func0(var0, var1, const0)
            ret1 = func1(ret0, const1)

            return ret1
    """

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
) -> Callable:
    """Compile the source code for a wrapper function into a Python function
    and return it.

    Any globals expected by the function need to be passed in the
    wrapper_globals dict.
    """
    my_globals = wrapper_globals.copy()
    exec(wrapper_text, my_globals)
    return my_globals[wrapper_name]


class NumbaCompiler(Compiler):
    def __init__(self, name="numba"):
        super().__init__(name=name)
        self._subgraph_count = 0

    def compile_algorithm(self, algo, literals):
        """Wrap a single function for JIT compilation and execution.
        
        literals is not used for anything currently
        """
        return numba.jit(algo.func)

    def compile_subgraph(
        self, subgraph: Dict, inputs: List[Hashable], output: Hashable
    ) -> Callable:
        """Fuse a subgraph of tasks into a single compiled function.

        It is assumed that the function will be called with values corresponding to
        `inputs` in the order they are given.
        """
        tbl = SymbolTable()

        # must populate the symbol table in toposort order
        toposort_keys = list(toposort(subgraph))

        # register the inputs as variables
        for key in inputs:
            tbl.register_var(key)

        # register each function in the subgraph
        for key in toposort_keys:
            task = subgraph[key]
            # all metagraph tasks are in (func, args, kwargs) format
            delayed_algo, args, kwargs = task
            if isinstance(kwargs, tuple):
                # FIXME: why are dictionaries represented this way in the DAG?
                kwargs = kwargs[0](kwargs[1])
            if len(kwargs) != 0:
                raise CompileError(
                    "NumbaCompiler only supports functions with bound kwargs.\n"
                    f"When compiling:\n{delayed_algo.func_label}\nfound unbound kwargs:\n{kwargs}"
                )
            # for maximum optimization, inline every task function during
            # compilation of the wrapper
            jit_func = numba.jit(inline="always")(delayed_algo.algo.func)
            tbl.register_func(key, jit_func, args)

        # generate the wrapper
        subgraph_wrapper_name = "subgraph" + str(self._subgraph_count)
        self._subgraph_count += 1
        wrapper_text, wrapper_globals = construct_call_wrapper_text(
            wrapper_name=subgraph_wrapper_name,
            symbol_table=tbl,
            input_keys=inputs,
            execute_keys=toposort_keys,
            output_key=output,
        )

        wrapper_func = compile_wrapper(
            subgraph_wrapper_name, wrapper_text, wrapper_globals
        )
        return wrapper_func
