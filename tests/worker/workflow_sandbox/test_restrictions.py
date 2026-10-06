from __future__ import annotations

import pathlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from temporalio.worker.workflow_sandbox._restrictions import (
    RestrictedModule,
    RestrictedWorkflowAccessError,
    RestrictionContext,
    SandboxMatcher,
    SandboxRestrictions,
    _RestrictedProxy,
)


def test_workflow_sandbox_restrictions_add_passthrough_modules():
    updated = SandboxRestrictions.default.with_passthrough_modules("module1", "module2")
    assert (
        "module1" in updated.passthrough_modules
        and "module2" in updated.passthrough_modules
    )


@dataclass
class RestrictableObject:
    foo: RestrictableObject | None = None
    bar: int = 42
    baz: ClassVar[int] = 57
    qux: ClassVar[RestrictableObject]

    some_dict: dict | None = None


RestrictableObject.qux = RestrictableObject(foo=RestrictableObject(bar=70), bar=80)


class RestrictableClass:
    def __str__(self):
        return "__str__"

    def __repr__(self):
        return "__repr__"

    def __format__(self, __format_spec: str) -> str:
        return "__format__"


def test_restricted_proxy_dunder_methods():
    restricted_class = _RestrictedProxy(
        "RestrictableClass",
        RestrictableClass,
        RestrictionContext(),
        SandboxMatcher(),
    )
    restricted_obj = restricted_class()
    assert type(restricted_obj) is _RestrictedProxy
    assert str(restricted_obj) == "__str__"
    assert repr(restricted_obj) == "__repr__"
    assert format(restricted_obj, "") == "__format__"
    assert f"{restricted_obj}" == "__format__"

    restricted_path = _RestrictedProxy(
        "Path",
        pathlib.Path,
        RestrictionContext(),
        SandboxMatcher(),
    )
    assert isinstance(format(restricted_path, ""), str)
    restricted_path_obj = restricted_path("test/path")
    assert type(restricted_path_obj) is _RestrictedProxy
    expected_path = str(pathlib.PurePath("test/path"))
    assert format(restricted_path_obj, "") == expected_path
    assert f"{restricted_path_obj}" == expected_path


def test_restricted_proxy_in_place_operations_preserve_proxy():
    restricted_list = _RestrictedProxy(
        "list",
        list,
        RestrictionContext(),
        SandboxMatcher(),
    )
    values = restricted_list([1])

    values += [2]

    assert type(values) is _RestrictedProxy
    assert RestrictionContext.unwrap_if_proxied(values) == [1, 2]


def test_workflow_sandbox_restricted_proxy():
    obj_class = _RestrictedProxy(
        "RestrictableObject",
        RestrictableObject,
        RestrictionContext(),
        SandboxMatcher(
            children={
                "foo": SandboxMatcher(access={"bar"}),
                "qux": SandboxMatcher(children={"foo": SandboxMatcher(access={"foo"})}),
                "some_dict": SandboxMatcher(
                    children={
                        "key1": SandboxMatcher(access={"subkey2"}),
                        "key.2": SandboxMatcher(access={"subkey2"}),
                    }
                ),
            }
        ),
    )
    obj = obj_class(
        foo=obj_class(),
        some_dict={
            "key1": {"subkey1": "val", "subkey2": "val"},
            "key.2": {"subkey1": "val", "subkey2": "val"},
        },
    )
    # Accessing these values is fine
    _ = (
        obj.bar,
        obj.foo.foo,
        obj_class.baz,
        obj_class.qux.bar,
        obj_class.qux.foo.bar,
        obj.some_dict["key1"]["subkey1"],
        obj.some_dict["key.2"]["subkey1"],
    )
    # But these aren't
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        _ = obj.foo.bar
    assert err.value.qualified_name == "RestrictableObject.foo.bar"
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        _ = obj_class.qux.foo.foo.bar
    assert err.value.qualified_name == "RestrictableObject.qux.foo.foo"
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        _ = obj.some_dict["key1"]["subkey2"]
    assert err.value.qualified_name == "RestrictableObject.some_dict.key1.subkey2"
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        _ = obj.some_dict["key.2"]["subkey2"]
    assert err.value.qualified_name == "RestrictableObject.some_dict.key.2.subkey2"

    # Unfortunately, we can't intercept the type() call
    assert type(obj.foo) is _RestrictedProxy


def _restricted_module(
    module: Any,
    context: RestrictionContext,
    *,
    invalid_members: SandboxMatcher | None = None,
) -> RestrictedModule:
    matcher = (
        invalid_members or SandboxRestrictions.invalid_module_members_default
    ).child_matcher(module.__name__)
    assert matcher
    return RestrictedModule(module, context, matcher)


def _context(*, is_runtime: bool = True) -> RestrictionContext:
    context = RestrictionContext()
    context.is_runtime = is_runtime
    return context


def test_matcher_allow_call_requires_match_self():
    with pytest.raises(ValueError, match="allow_call"):
        SandboxMatcher(allow_call=lambda: True)


def test_restricted_proxy_allow_call():
    def double(value: int | None = None) -> list[int]:
        return [] if value is None else [value * 2]

    func = _RestrictedProxy(
        "double",
        double,
        _context(),
        SandboxMatcher(
            children={
                "__call__": SandboxMatcher(
                    match_self=True,
                    allow_call=lambda *args, **kwargs: bool(args or kwargs),
                )
            }
        ),
    )
    # Allowed calls give back the plain result, not a proxy
    assert type(func(2)) is list
    assert func(value=3) == [6]
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        func()
    assert err.value.qualified_name == "double.__call__"


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.gmtime(),
        lambda t: t.gmtime(None),
        lambda t: t.ctime(),
        lambda t: t.ctime(None),
        lambda t: t.asctime(),
        lambda t: t.strftime("%Y"),
    ],
)
def test_restricted_time_calls_without_time_argument(
    call: Callable[[RestrictedModule], Any],
):
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        call(_restricted_module(time, _context()))
    assert "non-deterministic" in str(err.value)
    # These are only restricted once the workflow is running
    call(_restricted_module(time, _context(is_runtime=False)))


def test_restricted_time_calls_with_time_argument():
    restricted_time = _restricted_module(time, _context())
    struct = restricted_time.gmtime(0)
    # Result must stay a real struct_time because C functions require that
    assert type(struct) is time.struct_time
    assert restricted_time.gmtime(86400)[:3] == (1970, 1, 2)
    assert restricted_time.ctime(0) == time.ctime(0)
    assert restricted_time.asctime(struct) == time.asctime(struct)
    assert restricted_time.strftime("%Y-%m-%d", struct) == "1970-01-01"
    assert restricted_time.strftime("%Y-%m-%d", tuple(struct)) == "1970-01-01"


@pytest.mark.skipif(
    not hasattr(time, "clock_gettime"), reason="clock_gettime is unavailable"
)
def test_restricted_time_clock_gettime():
    restricted_time = _restricted_module(time, _context())
    # Referencing is fine, calling is not
    for name in ["clock_gettime", "clock_gettime_ns"]:
        with pytest.raises(RestrictedWorkflowAccessError) as err:
            getattr(restricted_time, name)(time.CLOCK_REALTIME)
        assert err.value.qualified_name == f"time.{name}.__call__"
    # Only restricted once the workflow is running
    _restricted_module(time, _context(is_runtime=False)).clock_gettime(0)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.Random(),
        lambda r: r.Random(None),
        lambda r: r.Random(x=None),
        lambda r: r.SystemRandom(),
    ],
)
def test_restricted_random_calls(call: Callable[[RestrictedModule], Any]):
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        call(_restricted_module(random, _context()))
    assert err.value.qualified_name.startswith("random.")
    assert "workflow.random()" in str(err.value)
    # These are only restricted once the workflow is running
    call(_restricted_module(random, _context(is_runtime=False)))


def test_restricted_random_with_seed():
    restricted_random = _restricted_module(random, _context())
    for seed in [0, 5, "seed", b"seed"]:
        # Instances are not proxied
        assert type(restricted_random.Random(seed)) is random.Random
    assert restricted_random.Random(x=5).random() == random.Random(5).random()
    assert isinstance(random.Random(5), restricted_random.Random)
    assert not isinstance(object(), restricted_random.Random)
    # Module-level functions share one global state and are still restricted
    with pytest.raises(RestrictedWorkflowAccessError):
        restricted_random.random()


def test_restricted_system_random_instance_created_at_import():
    context = _context(is_runtime=False)
    system_random = _restricted_module(random, context).SystemRandom()
    context.is_runtime = True
    with pytest.raises(RestrictedWorkflowAccessError) as err:
        system_random.random()
    assert err.value.qualified_name == "random.SystemRandom.random.__call__"
    assert not hasattr(random, "SecureRandom")


def test_restricted_call_forms_can_be_unrestricted():
    invalid_members = (
        SandboxRestrictions.invalid_module_members_default.with_child_unrestricted(
            "random", "Random"
        ).with_child_unrestricted("time", "gmtime")
    )
    restricted_random = _restricted_module(
        random, _context(), invalid_members=invalid_members
    )
    restricted_time = _restricted_module(
        time, _context(), invalid_members=invalid_members
    )
    restricted_random.Random()
    restricted_time.gmtime()
    with pytest.raises(RestrictedWorkflowAccessError):
        restricted_random.SystemRandom()
    with pytest.raises(RestrictedWorkflowAccessError):
        restricted_time.ctime()
