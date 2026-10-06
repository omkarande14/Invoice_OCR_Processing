# Known Limits: Static AST Leakage Screening

This document outlines structural limitations of static AST-based screening for data leakage and coordinate hardcoding.

## 1. Dynamic / Obfuscated Runtime String Construction
- **Character Array / Byte Manipulation**: Strings reconstructed dynamically via `chr()`, base64 encoding, hex decoding, or dynamic zlib decompression cannot be statically resolved by AST inspection alone.
  - *Mitigation*: Pre-commit hooks prohibit non-synthetic binary decoders and non-standard string decompression in pipeline modules.
- **Complex External Lookups**: If a candidate loads tokens dynamically from an untracked external JSON file or environment variable at runtime, static AST scanning of the code diff will not detect the strings directly.
  - *Mitigation*: CI environment runs with hermetic file system checks; pipeline modules are not allowed to load arbitrary external files outside `config/` or `data/`.

## 2. Indirect Mathematical Coordinate Offsets
- **Dynamic Variable Inversion**: Expressions where coordinate names are obscured or deeply nested through custom classes (e.g. `obj.geometry.points[0].matrix_mul(...)`) may evade attribute-name heuristics (`x`, `y`, `box`, `bbox`).
  - *Mitigation*: The OCR pipeline uses standard dictionary and tuple bounding box representations (`box`, `bbox`, `[ymin, xmin, ymax, xmax]`), which are all actively matched by `_is_coordinate_node()`.

## 3. Highly Transformed Numbers
- **Arithmetic Composition of Constants**: Numbers composed via arithmetic operations (e.g. `50000 + 4421` instead of `54421`) require full symbolic execution to resolve constant folding.
  - *Mitigation*: Constant-folded string concatenation (`'WMP/B/' + '04531'`) is explicitly resolved in AST `visit_BinOp`.
