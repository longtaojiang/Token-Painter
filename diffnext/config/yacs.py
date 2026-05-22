# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Yet Another Configuration System (YACS)."""

import copy

import numpy as np
import yaml


class CfgNode(dict):
    """Node for configuration options."""

    IMMUTABLE = "__immutable__"

    def __init__(self, *args, **kwargs):
        super(CfgNode, self).__init__(*args, **kwargs)
        self.__dict__[CfgNode.IMMUTABLE] = False

    def clone(self):
        """Recursively copy this CfgNode."""
        return copy.deepcopy(self)

    def freeze(self):
        """Make this CfgNode and all of its children immutable."""
        self._immutable(True)

    def is_frozen(self):
        """Return mutability."""
        return self.__dict__[CfgNode.IMMUTABLE]

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it into this CfgNode."""
        with open(cfg_filename, "r") as f:
            other_cfg = CfgNode(yaml.safe_load(f))
        self.merge_from_other_cfg(other_cfg)

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list into this CfgNode."""
        assert len(cfg_list) % 2 == 0
        from ast import literal_eval

        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = k.split(".")
            d = self
            for sub_key in key_list[:-1]:
                assert sub_key in d
                d = d[sub_key]
            sub_key = key_list[-1]
            assert sub_key in d
            try:
                value = literal_eval(v)
            except:  # noqa
                # Handle the case when v is a string literal
                value = v
            if type(value) != type(d[sub_key]):  # noqa
                raise TypeError(
                    "Type {} does not match original type {}".format(type(value), type(d[sub_key]))
                )
            d[sub_key] = value

    def merge_from_other_cfg(self, other_cfg):
        """Merge ``other_cfg`` into this CfgNode."""
        _merge_a_into_b(other_cfg, self)

    def _immutable(self, is_immutable):
        """Set immutability recursively to all nested CfgNode."""
        self.__dict__[CfgNode.IMMUTABLE] = is_immutable
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)
        for v in self.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())

    def __setattr__(self, name, value):
        if not self.__dict__[CfgNode.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but CfgNode is immutable'.format(name, value)
            )

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a."""
    if not isinstance(a, dict):
        return
    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))
        # The types must match, too
        v = _check_and_coerce_cfg_value_type(v, b[k], k)
        # Recursively merge dicts
        if type(v) is CfgNode:
            try:
                _merge_a_into_b(a[k], b[k])
            except:  # noqa
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def _check_and_coerce_cfg_value_type(value_a, value_b, key):
    """Check if the value type matched."""
    type_a, type_b = type(value_a), type(value_b)
    if type_a is type_b:
        return value_a
    if type_b is float and type_a is int:
        return float(value_a)
    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, dict) and isinstance(value_b, CfgNode):
        value_a = CfgNode(value_a)
    return value_a
