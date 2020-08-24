#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass
