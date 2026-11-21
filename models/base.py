"""
Registry for neural network models
"""

from abc import ABC, abstractmethod

import numpy as np

MODEL_REGISTRY = {}


def register_model(name: str):
  """Decorator: saves a neural network model under a name"""

  def deco(cls):
    MODEL_REGISTRY[name] = cls
    return cls
  
  return deco