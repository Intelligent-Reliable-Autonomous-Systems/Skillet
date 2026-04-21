"""Init file for skillet_tasks."""

# Imports to fix protobuf compatibility
import collections
import collections.abc

collections.MutableMapping = collections.abc.MutableMapping
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableSequence = collections.abc.MutableSequence
collections.Callable = collections.abc.Callable
