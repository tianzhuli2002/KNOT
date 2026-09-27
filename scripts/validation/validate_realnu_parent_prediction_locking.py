"""Check a closed parent prediction opens without unsupported filesystem locking."""

from __future__ import annotations

import errno
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py

from joint_event_reconstruction.prediction_partonic import _open_validation_file


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jer-hdf5-lock-retry-") as directory:
        path = Path(directory) / "prediction.h5"
        with h5py.File(path, "w") as output:
            output.create_dataset("EVENT_ID", data=[17, 23])

        real_open = h5py.File
        calls = 0

        def unsupported_locking(file, mode="r", *args, **kwargs):
            nonlocal calls
            if Path(file) == path and mode == "r":
                calls += 1
                if kwargs.get("locking") is not False:
                    raise BlockingIOError(errno.EAGAIN, "shared filesystem locking is unavailable")
            return real_open(file, mode, *args, **kwargs)

        with (
            patch.object(h5py, "File", side_effect=unsupported_locking),
            _open_validation_file(path) as source,
        ):
            observed = source["EVENT_ID"][:].tolist()

        if observed != [17, 23] or calls != 1:
            raise RuntimeError(f"HDF5 validation retry failed: events={observed}, opens={calls}")
    print("PASS: private closed prediction opens with HDF5 locking disabled")


if __name__ == "__main__":
    main()
