#!/usr/bin/env python3
"""
Embed shader binary into C++ header file.
"""

import sys
import os

def embed_shader(input_file, output_file, name):
    """Embed shader binary into C++ header."""

    # Read binary data
    if os.path.exists(input_file):
        with open(input_file, 'rb') as f:
            data = f.read()
    else:
        data = b''

    # Generate C++ header
    with open(output_file, 'w') as f:
        f.write("// Auto-generated shader header\n")
        f.write("// Do not edit manually\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n\n")
        f.write("namespace gaussian_rt {\n\n")

        # Write byte array
        f.write(f"static const uint8_t {name}_data[] = {{\n")
        for i, byte in enumerate(data):
            if i % 16 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x},")
            if i % 16 == 15:
                f.write("\n")
            else:
                f.write(" ")
        if len(data) % 16 != 0:
            f.write("\n")
        f.write("};\n\n")

        f.write(f"static const size_t {name}_size = {len(data)};\n\n")
        f.write("} // namespace gaussian_rt\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file> <name>")
        sys.exit(1)

    embed_shader(sys.argv[1], sys.argv[2], sys.argv[3])
