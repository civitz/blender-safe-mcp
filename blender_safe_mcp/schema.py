"""Tiny strict parameter validator. Pure Python, no bpy.

Each spec validates a JSON value and can describe itself as JSON Schema, so the
MCP tool schemas come from the addon's own specs.
"""

import math
import re


class ValidationError(Exception):
    def __init__(self, path, message):
        super().__init__(f"{path or 'params'}: {message}")
        self.path = path


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class Spec:
    description = None

    def validate(self, value, path=""):
        raise NotImplementedError

    def json_schema(self):
        raise NotImplementedError

    def _with_desc(self, schema):
        if self.description:
            schema["description"] = self.description
        return schema


class Int(Spec):
    def __init__(self, min=None, max=None, description=None):
        self.min, self.max, self.description = min, max, description

    def validate(self, value, path=""):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(path, "expected integer")
        if self.min is not None and value < self.min:
            raise ValidationError(path, f"must be >= {self.min}")
        if self.max is not None and value > self.max:
            raise ValidationError(path, f"must be <= {self.max}")
        return value

    def json_schema(self):
        s = {"type": "integer"}
        if self.min is not None:
            s["minimum"] = self.min
        if self.max is not None:
            s["maximum"] = self.max
        return self._with_desc(s)


class Float(Spec):
    def __init__(self, min=None, max=None, description=None):
        self.min, self.max, self.description = min, max, description

    def validate(self, value, path=""):
        if not _is_num(value) or not math.isfinite(value):
            raise ValidationError(path, "expected finite number")
        value = float(value)
        if self.min is not None and value < self.min:
            raise ValidationError(path, f"must be >= {self.min}")
        if self.max is not None and value > self.max:
            raise ValidationError(path, f"must be <= {self.max}")
        return value

    def json_schema(self):
        s = {"type": "number"}
        if self.min is not None:
            s["minimum"] = self.min
        if self.max is not None:
            s["maximum"] = self.max
        return self._with_desc(s)


class Bool(Spec):
    def __init__(self, description=None):
        self.description = description

    def validate(self, value, path=""):
        if not isinstance(value, bool):
            raise ValidationError(path, "expected boolean")
        return value

    def json_schema(self):
        return self._with_desc({"type": "boolean"})


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class Str(Spec):
    def __init__(self, max_len=64, pattern=None, min_len=1, multiline=False, description=None):
        self.max_len, self.min_len = max_len, min_len
        self.pattern = re.compile(pattern) if pattern else None
        self.multiline = multiline
        self.description = description

    def validate(self, value, path=""):
        if not isinstance(value, str):
            raise ValidationError(path, "expected string")
        if not self.min_len <= len(value) <= self.max_len:
            raise ValidationError(path, f"length must be {self.min_len}..{self.max_len}")
        if _CONTROL.search(value) or (not self.multiline and ("\n" in value or "\r" in value)):
            raise ValidationError(path, "control characters not allowed")
        if self.pattern and not self.pattern.fullmatch(value):
            raise ValidationError(path, f"must match {self.pattern.pattern}")
        return value

    def json_schema(self):
        s = {"type": "string", "minLength": self.min_len, "maxLength": self.max_len}
        if self.pattern:
            s["pattern"] = f"^(?:{self.pattern.pattern})$"
        return self._with_desc(s)


class Enum(Spec):
    def __init__(self, choices, description=None):
        self.choices = tuple(choices)
        self.description = description

    def validate(self, value, path=""):
        if not isinstance(value, str) or value not in self.choices:
            raise ValidationError(path, f"must be one of {list(self.choices)}")
        return value

    def json_schema(self):
        return self._with_desc({"type": "string", "enum": list(self.choices)})


class List(Spec):
    def __init__(self, item, min_len=0, max_len=64, description=None):
        self.item, self.min_len, self.max_len = item, min_len, max_len
        self.description = description

    def validate(self, value, path=""):
        if not isinstance(value, list):
            raise ValidationError(path, "expected array")
        if not self.min_len <= len(value) <= self.max_len:
            raise ValidationError(path, f"length must be {self.min_len}..{self.max_len}")
        return [self.item.validate(v, f"{path}[{i}]") for i, v in enumerate(value)]

    def json_schema(self):
        return self._with_desc({"type": "array", "items": self.item.json_schema(),
                                "minItems": self.min_len, "maxItems": self.max_len})


def Vec(n, min=None, max=None, description=None):
    return List(Float(min, max), min_len=n, max_len=n, description=description)


def Color(alpha=False, description=None):
    return Vec(4 if alpha else 3, 0.0, 1.0, description=description or
               ("RGBA" if alpha else "RGB") + " in 0..1 (linear)")


class Any(Spec):
    """Any JSON scalar or flat array of scalars (for prop_set values, checked later by RNA)."""

    def __init__(self, max_len=64, description=None):
        self.max_len, self.description = max_len, description

    def validate(self, value, path=""):
        def scalar(v, p):
            if isinstance(v, bool) or v is None:
                return v
            if _is_num(v):
                if not math.isfinite(v):
                    raise ValidationError(p, "non-finite number")
                return v
            if isinstance(v, str):
                return Str(max_len=4096, min_len=0, multiline=True).validate(v, p)
            raise ValidationError(p, "expected scalar")
        if isinstance(value, list):
            if len(value) > self.max_len:
                raise ValidationError(path, "array too long")
            return [scalar(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return scalar(value, path)

    def json_schema(self):
        scalar = {"type": ["number", "boolean", "string", "null"]}
        return self._with_desc({"anyOf": [scalar, {"type": "array", "items": scalar,
                                                   "maxItems": self.max_len}]})


class Union(Spec):
    def __init__(self, *options, description=None):
        self.options, self.description = options, description

    def validate(self, value, path=""):
        errors = []
        for o in self.options:
            try:
                return o.validate(value, path)
            except ValidationError as e:
                errors.append(str(e))
        raise ValidationError(path, "no alternative matched: " + " | ".join(errors))

    def json_schema(self):
        return self._with_desc({"anyOf": [o.json_schema() for o in self.options]})


_MISSING = object()


class Field:
    def __init__(self, spec, default=_MISSING, required=None):
        self.spec = spec
        self.default = default
        self.required = (default is _MISSING) if required is None else required


def Opt(spec, default=None):
    return Field(spec, default=default, required=False)


class Obj(Spec):
    """Object with a fixed set of keys. Unknown keys are rejected."""

    def __init__(self, fields=None, description=None):
        self.fields = {k: (f if isinstance(f, Field) else Field(f))
                       for k, f in (fields or {}).items()}
        self.description = description

    def validate(self, value, path=""):
        if not isinstance(value, dict):
            raise ValidationError(path, "expected object")
        unknown = set(value) - set(self.fields)
        if unknown:
            raise ValidationError(path, f"unknown keys {sorted(unknown)}")
        out = {}
        for k, f in self.fields.items():
            sub = f"{path}.{k}" if path else k
            if k in value and not (value[k] is None and not f.required):
                out[k] = f.spec.validate(value[k], sub)
            elif f.required:
                raise ValidationError(sub, "is required")
            else:
                out[k] = f.default
        return out

    def json_schema(self):
        props = {k: f.spec.json_schema() for k, f in self.fields.items()}
        s = {"type": "object", "properties": props, "additionalProperties": False}
        req = [k for k, f in self.fields.items() if f.required]
        if req:
            s["required"] = req
        return self._with_desc(s)
