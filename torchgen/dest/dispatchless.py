from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from torchgen.api import dispatchless, cpp
from torchgen.api.types import (
    CppSignatureGroup,
    kernel_signature,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
)

from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
    Variant,
)
from torchgen.utils import assert_never

# necessary information for an operation marked as dispatch-less composite.
@dataclass(frozen=True)
class CompositeInfo:
    # the actual NativeFunction that corresponds to the OperatorName.
    f: NativeFunction

    # a list of dependencies.
    # for each dependency, we also save its NativeFunctionsGroup (if there
    # is one) because of how structured kernels are generated.
    # if a NativeFunction has no BackendMetadata, it can still be registered
    # to a dispatch key if it's structured (i.e. its out function is structured).
    dependencies: List[Tuple[NativeFunction, Optional[NativeFunctionsGroup]]]


# Dependency information for dispatch-less composite kernels.
# Maps the overloaded operator name to a list of operations it depends on.
# We need to store both the 'NativeFunction' it depends on, as well as the
# 'NativeFunctionsGroup' it belongs to in order to find out whether the
# operation has a kernel registered for a given dispatch key.
CompositeGraph = Dict[OperatorName, CompositeInfo]


def get_native_and_group_pair(
    g: Union[NativeFunction, NativeFunctionsGroup]
) -> List[Tuple[NativeFunction, Optional[NativeFunctionsGroup]]]:
    if isinstance(g, NativeFunction):
        fs = [g]
        group = None
    elif isinstance(g, NativeFunctionsGroup):
        fs = list(g.functions())
        group = g
    else:
        assert_never(g)
    return [(f, group) for f in fs]


def get_graph(
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    b: BackendIndex,
) -> CompositeGraph:
    opname_to_native_map = {
        f.func.name: (f, group)
        for g in grouped_native_functions
        for f, group in get_native_and_group_pair(g)
    }

    return {
        f.func.name: CompositeInfo(
            f,
            sorted(
                (opname_to_native_map[opname] for opname in f.composite[0]), key=str
            ),
        )
        for g in grouped_native_functions
        for f, _ in get_native_and_group_pair(g)
        if len(f.composite[0]) > 0 and b.dispatch_key in f.composite[1]
    }


# convenience function for checking if a NativeFunction is registered to a
# dispatch key.
def has_registered_kernel(
    index: BackendIndex, f: NativeFunction, g: Optional[NativeFunctionsGroup]
) -> bool:
    if g is not None and g.structured:
        # if f is structured, then it will be registered for both Meta and
        # CompositeExpicitAutograd (both not in STRUCTURED_DISPATCH_KEYS).
        cond_meta = index.dispatch_key == DispatchKey.Meta

        # structured kernel generation does not register out functions for
        # CompositeExplicitAutograd.
        cond_composite = (
            index.dispatch_key == DispatchKey.CompositeExplicitAutograd
            and not f.func.is_out_fn()
        )
        return index.has_kernel(g) or cond_meta or cond_composite
    return index.has_kernel(f)


@dataclass(frozen=True)
class DispatchlessComposite:
    # dispatch key we are generating for.
    dispatch_key: DispatchKey

    # we need all backend indices, since we will iterate a specific order
    # of dispatch keys looking for registrations.
    backend_indices: Dict[DispatchKey, BackendIndex]

    # a graph of dependency relations between operators.
    graph: CompositeGraph

    @staticmethod
    def new(
        dispatch_key: DispatchKey,
        backend_indices: Dict[DispatchKey, BackendIndex],
        grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    ) -> "DispatchlessComposite":
        graph = get_graph(grouped_native_functions, backend_indices[dispatch_key])
        return DispatchlessComposite(dispatch_key, backend_indices, graph)

    @property
    def backend_index(self) -> BackendIndex:
        return self.backend_indices[self.dispatch_key]

    # figures out what namespace to dispatch.
    def _dispatch_namespace(
        self, f: NativeFunction, g: Optional[NativeFunctionsGroup]
    ) -> Optional[DispatchKey]:
        # the precedence order is:
        #   1. the current dispatch key we are generating code for
        #   2. the CompositeExplicitAutograd kernel
        #   3. the CompositeImplicitAutograd kernel
        for k in (
            self.dispatch_key,
            DispatchKey.CompositeExplicitAutograd,
            DispatchKey.CompositeImplicitAutograd,
        ):
            if has_registered_kernel(self.backend_indices[k], f, g):
                return k

        # bail if we have found no registered kernel.
        return None

    def _header_set(self, header_from_fn: Callable) -> List[str]:
        return list(
            {
                header_from_fn(dep_f, dep_g)
                for _, info in self.graph.items()
                for dep_f, dep_g in info.dependencies
            }
        )

    def aggregated_headers(self) -> List[str]:
        def header_from(
            dep_f: NativeFunction, dep_g: Optional[NativeFunctionsGroup]
        ) -> str:
            ns = self._dispatch_namespace(dep_f, dep_g)
            prefix = f"{ns}" if ns is not None else ""
            return f"#include <ATen/{prefix}Functions.h>"

        return self._header_set(header_from)

    def operator_headers(self) -> List[str]:
        def header_from(
            dep_f: NativeFunction, dep_g: Optional[NativeFunctionsGroup]
        ) -> str:
            ns = self._dispatch_namespace(dep_f, dep_g)
            suffix = f"_{ns.lower()}_dispatch" if ns is not None else ""
            return f"#include <ATen/ops/{dep_f.root_name}{suffix}.h>"

        return self._header_set(header_from)

    def headers(self) -> List[str]:
        native_functions = [info.f for _, info in self.graph.items()]
        return sorted(
            {
                f"#include <ATen/native/composite/{f.root_name}.h>"
                for f in native_functions
            }
        )

    @method_with_native_function
    def definition(self, f: NativeFunction) -> str:
        if f.func.name not in self.graph:
            return ""

        # define the native function as a wrapper static method, redirecting the call to
        # the appropriate function (e.g. on CPU, on CUDA, or as a CompositeExplicitAutograd).
        def dispatchless_function_defn(
            dep_f: NativeFunction, dep_g: Optional[NativeFunctionsGroup]
        ) -> str:
            with native_function_manager(dep_f):
                ns = self._dispatch_namespace(dep_f, dep_g)
                if ns is None:
                    # if we haven't found a namespace, we defer it to the dispatcher.
                    # however, there's a possibility that such an operation is only available
                    # as a tensor method.
                    if Variant.function in dep_f.variants:
                        prefix = "at::"
                        method = False
                    else:
                        assert Variant.method in dep_f.variants
                        assert dep_f.func.arguments.self_arg is not None
                        prefix = f"{dep_f.func.arguments.self_arg.argument.name}."
                        method = True
                else:
                    prefix = f"at::{ns.lower()}::"
                    method = False

                # the declaration signature should be as one would expect of the
                # C++ API function.
                decl_sig = CppSignatureGroup.from_native_function(
                    dep_f, method=False, fallback_binding=False
                ).signature

                # the call signature should reflect whether we are calling a tensor
                # method or not.
                call_sig = CppSignatureGroup.from_native_function(
                    dep_f, method=method, fallback_binding=False
                ).signature

                args_str = ", ".join(a.name for a in call_sig.arguments())
                return f"""
  static {decl_sig.decl()} {{
    return {prefix}{cpp.name(dep_f.func)}({args_str});
  }}"""

        # definition of each dependent operation.
        composite_definition_s = "\n".join(
            dispatchless_function_defn(cf, cg)
            for cf, cg in self.graph[f.func.name].dependencies
        )

        # definition of the struct.
        impl_struct = f"""
struct {dispatchless.struct(f)} {{
{composite_definition_s}
}};
"""

        # definition of the native function.
        # we match the signature to the one used when declaring it:
        # (at 'dest/native_functions.py')
        sig = kernel_signature(f, self.backend_index)
        args_str = ", ".join(a.name for a in sig.arguments())

        # put the struct definition in an anonymous namespace (avoiding name
        # clash), and the definition of the wrapper native function in the
        # 'at::native' namespace.
        return f"""
// {f.func}
namespace {{
{impl_struct}
}} // anonymous namespace

{sig.defn(name=dispatchless.kernel(f.func, self.dispatch_key))} {{
  return {dispatchless.call(f)}({args_str});
}}

"""