"""Check retry of a transient HDF5 EAGAIN while reopening a closed prediction."""

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

        def transient_lock(file, *args, **kwargs):
            nonlocal calls
            if Path(file) == path and args == ("r",) and calls == 0:
                calls += 1
                raise BlockingIOError(errno.EAGAIN, "transient HDF5 lock contention")
            calls += 1
            return real_open(file, *args, **kwargs)

        with (
            patch.object(h5py, "File", side_effect=transient_lock),
            _open_validation_file(path) as source,
        ):
            observed = source["EVENT_ID"][:].tolist()

        if observed != [17, 23] or calls != 2:
            raise RuntimeError(f"HDF5 validation retry failed: events={observed}, opens={calls}")
    print("PASS: transient HDF5 EAGAIN retried and prediction reopened")


if __name__ == "__main__":
    main()
