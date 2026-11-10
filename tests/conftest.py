"""Minimal rclpy and interface stand-ins, so cage node wiring runs without ROS 2.

Only what the cage nodes touch is modelled. Three rclpy behaviours are copied on
purpose: Node.__init__ declares ``use_sim_time`` (rclpy's TimeSource does),
declaring a parameter twice raises ParameterAlreadyDeclaredException, and a
backward clock jump (Ros.rewind) runs the clock's jump callbacks.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import ClassVar

import pytest


class ParameterAlreadyDeclaredException(Exception):
    pass


class Msg:
    """Attribute bag standing in for a generated message. Nested fields appear on
    first access so ``msg.header.stamp = ...`` works; an unset leaf is not a value."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)

    def __getattr__(self, name: str) -> Msg:
        if name.startswith("__"):
            raise AttributeError(name)
        value = Msg()
        setattr(self, name, value)
        return value

    def __bool__(self) -> bool:
        raise TypeError("unset message field used as a value")


def stamp(t: float) -> Msg:
    ns = round(t * 1e9)
    return Msg(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class _Time:
    def __init__(self, ns: int) -> None:
        self.nanoseconds = ns

    def to_msg(self) -> Msg:
        return stamp(self.nanoseconds * 1e-9)


class Duration:
    def __init__(self, *, seconds: float = 0, nanoseconds: int = 0) -> None:
        self.nanoseconds = round(seconds * 1e9) + nanoseconds


class JumpThreshold:
    def __init__(self, *, min_forward: Duration | None, min_backward: Duration | None, on_clock_change: bool = True):
        self.min_forward = min_forward
        self.min_backward = min_backward
        self.on_clock_change = on_clock_change


class _Param:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self.value = value


class Node:
    overrides: ClassVar[dict[str, object]] = {}

    def __init__(self, node_name: str, **_kwargs: object) -> None:
        self.node_name = node_name
        self.now_ns = 0
        self.params: dict[str, _Param] = {}
        self.subs: dict[str, list] = {}
        self.sent: dict[str, list] = {}
        self.services: dict[str, object] = {}
        self.timers: list = []
        self.logs: list[tuple[str, str]] = []
        self.jump_callbacks: list = []
        self.declare_parameter("use_sim_time", False)

    def declare_parameter(self, name: str, value: object = None, *_a: object, **_k: object) -> _Param:
        if name in self.params:
            raise ParameterAlreadyDeclaredException(name)
        self.params[name] = _Param(name, Node.overrides.get(name, value))
        return self.params[name]

    def get_parameter(self, name: str) -> _Param:
        return self.params[name]

    def create_publisher(self, _type: object, topic: str, _qos: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(publish=self.sent.setdefault(topic, []).append)

    def create_subscription(self, _type: object, topic: str, callback, _qos: object) -> None:
        self.subs.setdefault(topic, []).append(callback)

    def create_service(self, _type: object, name: str, callback) -> None:
        self.services[name] = callback

    def create_timer(self, _period: float, callback) -> None:
        self.timers.append(callback)

    def get_clock(self) -> types.SimpleNamespace:
        def create_jump_callback(threshold: JumpThreshold, *, pre_callback=None, post_callback=None) -> object:
            self.jump_callbacks.append((threshold, post_callback))
            return object()

        return types.SimpleNamespace(now=lambda: _Time(self.now_ns), create_jump_callback=create_jump_callback)

    def get_logger(self) -> types.SimpleNamespace:
        def log(level: str):
            return lambda text, *_a, **_k: self.logs.append((level, str(text)))

        return types.SimpleNamespace(**{lv: log(lv) for lv in ("debug", "info", "warn", "warning", "error", "fatal")})

    def destroy_node(self) -> None:
        pass


def _interfaces(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    made: dict[str, type] = {}

    def __getattr__(attr: str) -> type:
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr not in made:
            cls = type(attr, (Msg,), {})
            cls.Request = type(f"{attr}_Request", (Msg,), {})
            cls.Response = type(f"{attr}_Response", (Msg,), {})
            made[attr] = cls
        return made[attr]

    module.__getattr__ = __getattr__
    return module


def _fake_modules() -> dict[str, types.ModuleType]:
    rclpy = types.ModuleType("rclpy")
    rclpy.init = rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = Node
    rclpy.exceptions = types.ModuleType("rclpy.exceptions")
    rclpy.exceptions.ParameterAlreadyDeclaredException = ParameterAlreadyDeclaredException
    rclpy.clock = types.ModuleType("rclpy.clock")
    rclpy.clock.JumpThreshold = JumpThreshold
    rclpy.duration = types.ModuleType("rclpy.duration")
    rclpy.duration.Duration = Duration
    modules = {
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "rclpy.exceptions": rclpy.exceptions,
        "rclpy.clock": rclpy.clock,
        "rclpy.duration": rclpy.duration,
    }
    for pkg, subs in (
        ("autocar_msgs", ("msg", "srv")),
        ("std_srvs", ("srv",)),
        ("geometry_msgs", ("msg",)),
        ("sensor_msgs", ("msg",)),
    ):
        parent = types.ModuleType(pkg)
        modules[pkg] = parent
        for sub in subs:
            child = _interfaces(f"{pkg}.{sub}")
            setattr(parent, sub, child)
            modules[f"{pkg}.{sub}"] = child
    return modules


class Ros:
    Msg = Msg
    stamp = staticmethod(stamp)

    @staticmethod
    def make(module: str, cls: str, **params: object) -> Node:
        Node.overrides = params
        try:
            return getattr(importlib.import_module(module), cls)()
        finally:
            Node.overrides = {}

    @staticmethod
    def deliver(node: Node, topic: str, msg: Msg) -> None:
        for callback in node.subs[topic]:
            callback(msg)

    @staticmethod
    def spin_once(node: Node, t: float) -> None:
        node.now_ns = round(t * 1e9)
        for callback in node.timers:
            callback()

    @staticmethod
    def rewind(node: Node, t: float) -> None:
        """/clock jumps back to t: rcl runs the backward-jump callbacks, and no
        node callback or timer reads the clock while it is there."""
        delta = round(t * 1e9) - node.now_ns
        assert delta < 0
        for threshold, callback in node.jump_callbacks:
            if threshold.min_backward is not None and delta <= threshold.min_backward.nanoseconds and callback:
                callback(Msg(clock_change=0, delta=Duration(nanoseconds=delta)))
        node.now_ns += delta

    @staticmethod
    def last(node: Node, topic: str) -> Msg:
        return node.sent[topic][-1]


@pytest.fixture
def ros(monkeypatch: pytest.MonkeyPatch):
    for name, module in _fake_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    yield Ros()
    for name in [n for n in sys.modules if n.startswith("autocar_") and n.endswith("_node")]:
        del sys.modules[name]
