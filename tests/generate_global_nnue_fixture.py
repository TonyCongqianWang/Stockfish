"""Generate a deterministic, UNTRAINED NNUE fixture using only the stdlib.

This is solely for format/loading/activation tests, never for playing-strength
measurements or distribution as the default evaluation network.
"""

import argparse
from array import array
import struct
import sys


def feature_hash():
    value = 0
    for component in (0x4F535031, 0x45535031, 0x47415432, 0x50465032, 0x50464332, 0x32B5E284):
        value = ((value << 1) | (value >> 31)) & 0xFFFFFFFF
        value ^= component
    return value


def fc_hash(l1):
    value = 0xEC42E90D ^ (l1 * 2)
    for outputs in (32, 32, 1):
        next_value = (0xCC03DAE4 + outputs) ^ (value >> 1) ^ ((value << 31) & 0xFFFFFFFF)
        value = (next_value + (0x538D24C7 if outputs != 1 else 0)) & 0xFFFFFFFF
    return value


def write_fixture(path, l1=1024):
    def integers(values, code):
        result = array(code, values)
        if sys.byteorder != "little":
            result.byteswap()
        return result.tobytes()

    with open(path, "wb") as out:
        desc = b"UNTRAINED global NNUE test fixture; not for playing chess."
        fh, fc = feature_hash(), fc_hash(l1)
        out.write(struct.pack("<III", 0x6A448AFC, fc ^ fh ^ (l1 * 2), len(desc)))
        out.write(desc)
        out.write(struct.pack("<I", fh ^ (l1 * 2)))
        out.write(integers((64 + j % 7 for j in range(l1)), "h"))
        # Same nonzero periodic weights as the independent C++ reference sums.
        for family, rows, width in ((0, 14952, l1*3//4), (3, 14952, l1//4),
                                    (1, 29904 + 16807 + 480, l1), (2, 45056, l1)):
            block = bytes(((row * 17 + j * 3 + family * 11) % 7 - 3) & 255
                          for row in range(7) for j in range(width))
            for _ in range(rows // 7):
                out.write(block)
            out.write(block[: (rows % 7) * width])
        for bucket in range(32):
            out.write(struct.pack("<I", fc))
            for outputs, inputs in ((32, l1), (32, 64), (1, 128)):
                out.write(integers([0] * outputs, "i"))
                out.write(bytes(((j + bucket) % 3 - 1) & 255 for j in range(outputs * inputs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    write_fixture(args.output)
