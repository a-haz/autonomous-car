"""perc_source:=gt|trt behind one tracker ABI (PR-07a).

gt wraps autocar_perc_gt. trt is the PR-07b detector and is not in this
build yet: selecting it fails loudly and never returns detections.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .tracked import TrackedObjects

PERC_SOURCES = ("gt", "trt")
TRT_UNAVAILABLE = "perc_source trt is not available in this build (detector lands in PR-07b)"


class PercSourceUnavailable(RuntimeError):
    pass


class PercBackend(Protocol):
    def detect(self, stamp_ns: int) -> TrackedObjects: ...


def validate_perc_source(raw: str) -> str:
    perc = str(raw).strip().lower()
    if perc not in PERC_SOURCES:
        raise ValueError(f"perc_source must be gt|trt, got {raw!r}")
    return perc


class GtSource:
    """Ground-truth objects through autocar_perc_gt, then into the ABI."""

    def __init__(self, objects: Callable[[], list]) -> None:
        from autocar_perc_gt.stub import to_tracked

        self._to_tracked = to_tracked
        self.objects = objects

    def detect(self, stamp_ns: int) -> TrackedObjects:
        return TrackedObjects.from_msg(self._to_tracked(stamp_ns, self.objects()))


class TrtSource:
    """Placeholder. Raises on construction and on use, never fakes output."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise PercSourceUnavailable(TRT_UNAVAILABLE)

    def detect(self, stamp_ns: int) -> TrackedObjects:
        raise PercSourceUnavailable(TRT_UNAVAILABLE)


def make_backend(perc_source: str, gt_objects: Callable[[], list] | None = None) -> PercBackend:
    perc = validate_perc_source(perc_source)
    if perc == "trt":
        return TrtSource()
    if gt_objects is None:
        raise ValueError("perc_source gt needs a ground-truth object source")
    return GtSource(gt_objects)


class Perception:
    """Front end: one backend in, latest /perception/tracked_objects out."""

    def __init__(self, perc_source: str, gt_objects: Callable[[], list] | None = None) -> None:
        self.perc_source = validate_perc_source(perc_source)
        self.backend = make_backend(self.perc_source, gt_objects)
        self.tracked: TrackedObjects | None = None

    def step(self, stamp_ns: int) -> TrackedObjects:
        self.tracked = self.backend.detect(stamp_ns)
        return self.tracked

    def msg(self) -> dict:
        if self.tracked is None:
            raise RuntimeError("no tracked objects yet, call step() first")
        return self.tracked.to_msg()
