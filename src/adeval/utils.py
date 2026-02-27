from typing import Optional, Dict, Any, Tuple, Callable, Union
import numpy as np
from multiprocessing.reduction import ForkingPickler
from types import CodeType, FunctionType, CellType, ModuleType
import marshal
import sys
from platform import python_implementation


if sys.version_info >= (3, 8) and python_implementation() == 'CPython':

    HAS_MP_WITH_LOCALS = True

    def _rebuild_code(state: bytes) -> CodeType:
        ret = marshal.loads(state)
        assert isinstance(ret, CodeType)
        return ret


    def _reduce_code(code: CodeType) -> Tuple[Callable, Tuple[bytes]]:
        return _rebuild_code, (marshal.dumps(code),)


    def _rebuild_module(fullname: str) -> ModuleType:
        from importlib import import_module
        return import_module(fullname)


    def _reduce_module(mod: ModuleType) -> Tuple[Callable, Tuple[str]]:
        from pickle import PickleError
        fullname = mod.__name__
        if fullname not in sys.modules:
            raise PickleError(f'module {fullname} is invalid')
        return _rebuild_module, (fullname,)


    def _rebuild_function(code: CodeType, global_vals: Dict[str, Any], cell_vals: Tuple):
        global_vals = global_vals.copy()
        global_vals['__builtins__'] = __builtins__
        return FunctionType(code, global_vals, code.co_name, closure=tuple(
            CellType(val) for val in cell_vals
        ))

    def _reduce_function(func: FunctionType) -> Union[
        Tuple[Callable, Tuple[bytes]],
        Tuple[Callable, Tuple[CodeType, Dict[str, Any], Tuple]]
    ]:

        if (func.__closure__ is not None and
            len(func.__closure__) > 0) or '<locals>' in repr(func):
            return _rebuild_function, (func.__code__, {
                name: val for name, val in func.__globals__.items()
                if name in func.__code__.co_names
            }, tuple(
                val.cell_contents for val in func.__closure__
            ) if func.__closure__ is not None else ())

        elif func.__module__ == '__main__':
            return _rebuild_function, (func.__code__, {
                name: val for name, val in func.__globals__.items()
                if name in func.__code__.co_names
            }, tuple()) 

        else:
            return NotImplemented


    def _override_FunctionType(self, obj):
        if isinstance(obj, FunctionType):
            return _reduce_function(obj)
        else:
            return NotImplemented

    ForkingPickler.reducer_override = _override_FunctionType
    ForkingPickler.register(CodeType, _reduce_code)
    ForkingPickler.register(ModuleType, _reduce_module)

else:
    HAS_MP_WITH_LOCALS = False

