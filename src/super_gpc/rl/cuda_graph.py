"""Replay a function of fixed-shape GPU tensors from a CUDA graph.

CoLA collection launches thousands of tiny kernels per step from Python
(forward kinematics, the sampled policy step); the host, not the GPU, limits
the step time. A captured graph replays the same kernels on the same inputs.
"""

from __future__ import annotations

import dataclasses

import torch


def _clone(out):
  if isinstance(out, torch.Tensor):
    return out.clone()
  if isinstance(out, tuple):
    return tuple(_clone(o) for o in out)
  if dataclasses.is_dataclass(out):
    fields = {f.name: _clone(getattr(out, f.name)) for f in dataclasses.fields(out)}
    return dataclasses.replace(out, **fields)
  return out


class CudaGraphed:
  """Call ``fn(*tensors, **static)`` by replaying a graph captured once per
  argument shapes and static keyword values. ``fn`` may read other tensors
  only if they are updated in place (parameters, buffers), never replaced.
  Outputs are cloned, unless ``clone`` is false: then the next call with the
  same key overwrites them. Arguments off CUDA call ``fn`` directly."""

  def __init__(self, fn, clone: bool = True):
    self.fn, self.clone = fn, clone
    self.graphs: dict = {}

  def __call__(self, *args: torch.Tensor, **static):
    if not args[0].is_cuda:
      return self.fn(*args, **static)
    key = (tuple(a.shape for a in args), tuple(sorted(static.items())))
    if key not in self.graphs:
      inputs = [a.clone() for a in args]
      current = torch.cuda.current_stream(args[0].device)
      side = torch.cuda.Stream(args[0].device)
      side.wait_stream(current)
      with torch.cuda.stream(side):  # warm up (cuBLAS handles, compiled kernels) outside capture
        self.fn(*inputs, **static)
      current.wait_stream(side)
      graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
        out = self.fn(*inputs, **static)
      self.graphs[key] = (graph, inputs, out)
    graph, inputs, out = self.graphs[key]
    for dst, src in zip(inputs, args, strict=True):
      dst.copy_(src)
    graph.replay()
    return _clone(out) if self.clone else out
