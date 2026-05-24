"""
File input/output for BioniumX objects.
"""
import h5py
import numpy as np


def write_object(obj, filename: str, fmt: str = "hdf5"):
    """
    Write a BioniumXObject to a file.
    """
    if fmt == "hdf5":
        with h5py.File(filename, "w") as f:
            f.attrs["class_name"] = obj.__class__.__name__
            for attr in obj._required_attrs:
                f.create_dataset(attr, data=getattr(obj, attr))
            if hasattr(obj, "err"):
                f.create_dataset("err", data=obj.err)

            meta_group = f.create_group("meta")
            for k, v in obj.meta.items():
                if v is not None:
                    meta_group.attrs[k] = v
    else:
        raise NotImplementedError(f"Format {fmt} not yet implemented.")


def read_object(cls, filename: str, fmt: str = "hdf5"):
    """
    Read a BioniumXObject from a file.
    """
    if fmt == "hdf5":
        with h5py.File(filename, "r") as f:
            kwargs = {}
            for attr in cls._required_attrs:
                kwargs[attr] = f[attr][:]
            if "err" in f:
                kwargs["err"] = f["err"][:]

            if "meta" in f:
                for k, v in f["meta"].attrs.items():
                    kwargs[k] = v

            return cls(**kwargs)
    else:
        raise NotImplementedError(f"Format {fmt} not yet implemented.")
