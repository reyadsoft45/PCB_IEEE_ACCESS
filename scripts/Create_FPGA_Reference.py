import cv2
import numpy as np
from pathlib import Path

INPUT = Path(
    r"C:\PCB_IEEE_ACCESS\fpga_test\input_64x64.png"
)

OUTPUT_DIR = Path(
    r"C:\PCB_IEEE_ACCESS\fpga_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Read grayscale
# ---------------------------------------------------------

img = cv2.imread(
    str(INPUT),
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise FileNotFoundError(INPUT)

# Force 64x64 for initial FPGA verification
img = cv2.resize(
    img,
    (64, 64),
    interpolation=cv2.INTER_AREA
)

# ---------------------------------------------------------
# FPGA-equivalent 5-point Laplacian sharpening
#
# out = 5*C - N - S - W - E
# clip to unsigned 8-bit
# ---------------------------------------------------------

p = np.pad(
    img.astype(np.int16),
    ((1, 1), (1, 1)),
    mode="edge"
)

center = p[1:-1, 1:-1]
top    = p[:-2, 1:-1]
bottom = p[2:, 1:-1]
left   = p[1:-1, :-2]
right  = p[1:-1, 2:]

result = (
    5 * center
    - top
    - bottom
    - left
    - right
)

result = np.clip(
    result,
    0,
    255
).astype(np.uint8)

# ---------------------------------------------------------
# Save images
# ---------------------------------------------------------

cv2.imwrite(
    str(OUTPUT_DIR / "input_64x64_gray.png"),
    img
)

cv2.imwrite(
    str(OUTPUT_DIR / "software_reference.png"),
    result
)

# FPGA input stream
img.flatten().tofile(
    OUTPUT_DIR / "input_pixels.bin"
)

# Expected FPGA output stream
result.flatten().tofile(
    OUTPUT_DIR / "software_reference.bin"
)

# Human-readable text
np.savetxt(
    OUTPUT_DIR / "input_pixels.txt",
    img.flatten(),
    fmt="%d"
)

np.savetxt(
    OUTPUT_DIR / "software_reference.txt",
    result.flatten(),
    fmt="%d"
)

print("=" * 60)
print("FPGA SOFTWARE REFERENCE CREATED")
print("=" * 60)

print("Input shape:", img.shape)
print("Pixels:", img.size)
print("Min:", result.min())
print("Max:", result.max())

print("\nFiles:")
print("input_pixels.bin")
print("software_reference.bin")
print("input_pixels.txt")
print("software_reference.txt")
print("software_reference.png")