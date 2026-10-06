import serial
import time
import numpy as np
import cv2
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

PORT = "COM6"
BAUD = 115200

WIDTH = 64
HEIGHT = 64
NUM_PIXELS = WIDTH * HEIGHT

ROOT = Path(r"C:\PCB_IEEE_ACCESS\fpga_test")

INPUT_FILE = ROOT / "input_pixels.bin"
REFERENCE_FILE = ROOT / "software_reference.bin"

HARDWARE_BIN = ROOT / "hardware_output.bin"
HARDWARE_IMG = ROOT / "hardware_output.png"
DIFF_IMG = ROOT / "difference.png"
REPORT_FILE = ROOT / "fpga_comparison_report.txt"


# ============================================================
# CHECK FILES
# ============================================================

assert INPUT_FILE.exists(), f"Missing: {INPUT_FILE}"
assert REFERENCE_FILE.exists(), f"Missing: {REFERENCE_FILE}"


input_pixels = np.fromfile(
    INPUT_FILE,
    dtype=np.uint8
)

reference = np.fromfile(
    REFERENCE_FILE,
    dtype=np.uint8
)


assert len(input_pixels) == NUM_PIXELS, (
    f"Input expected {NUM_PIXELS}, got {len(input_pixels)}"
)

assert len(reference) == NUM_PIXELS, (
    f"Reference expected {NUM_PIXELS}, got {len(reference)}"
)


print("=" * 65)
print("FPGA UART IMAGE VERIFICATION")
print("=" * 65)

print("Port:", PORT)
print("Baud:", BAUD)
print("Image:", f"{WIDTH}x{HEIGHT}")
print("Pixels:", NUM_PIXELS)


# ============================================================
# OPEN UART
# ============================================================

  ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=2.0,
    write_timeout=2.0
)

time.sleep(2)

# Remove any old UART data
ser.reset_input_buffer()
ser.reset_output_buffer()

print("\nUART opened successfully.")


# ============================================================
# SEND INPUT IMAGE
# ============================================================

print("\nSending 4096 pixels to FPGA...")

start_time = time.perf_counter()

bytes_sent = ser.write(
    input_pixels.tobytes()
)

ser.flush()

print("Bytes sent:", bytes_sent)


# ============================================================
# RECEIVE FPGA OUTPUT
# ============================================================

print("\nWaiting for FPGA output...")

received = bytearray()

timeout_start = time.time()

while len(received) < NUM_PIXELS:

    available = ser.in_waiting

    if available > 0:

        remaining = NUM_PIXELS - len(received)

        chunk = ser.read(
            min(available, remaining)
        )

        received.extend(chunk)

    # Overall safety timeout
    if time.time() - timeout_start > 15:

        print("\nTIMEOUT!")
        break


end_time = time.perf_counter()

ser.close()


# ============================================================
# CHECK RECEIVED DATA
# ============================================================

print("\nBytes received:", len(received))

if len(received) != NUM_PIXELS:

    print("\nERROR:")
    print(
        f"Expected {NUM_PIXELS} bytes "
        f"but received {len(received)}."
    )

    print(
        "\nDo not continue to YOLO/FPGA comparison yet."
    )

    raise RuntimeError(
        "Incomplete FPGA output"
    )


hardware = np.frombuffer(
    received,
    dtype=np.uint8
).copy()

hardware.tofile(
    HARDWARE_BIN
)


# ============================================================
# RESHAPE
# ============================================================

ref_img = reference.reshape(
    HEIGHT,
    WIDTH
)

hw_img = hardware.reshape(
    HEIGHT,
    WIDTH
)


# ============================================================
# FULL IMAGE COMPARISON
# ============================================================

diff = np.abs(
    ref_img.astype(np.int16)
    -
    hw_img.astype(np.int16)
)

max_diff = int(diff.max())
mean_diff = float(diff.mean())

exact_match = int(
    np.sum(diff == 0)
)

mae = float(
    np.mean(diff)
)

rmse = float(
    np.sqrt(
        np.mean(diff.astype(np.float32) ** 2)
    )
)


# ============================================================
# INTERIOR 62x62 COMPARISON
# Ignore border for FPGA-border differences
# ============================================================

ref_inner = ref_img[1:-1, 1:-1]
hw_inner = hw_img[1:-1, 1:-1]

inner_diff = np.abs(
    ref_inner.astype(np.int16)
    -
    hw_inner.astype(np.int16)
)

inner_max = int(
    inner_diff.max()
)

inner_mean = float(
    inner_diff.mean()
)

inner_exact = int(
    np.sum(inner_diff == 0)
)

inner_total = inner_diff.size


# ============================================================
# SAVE IMAGES
# ============================================================

cv2.imwrite(
    str(HARDWARE_IMG),
    hw_img
)

# Scale difference image for easier visualization
diff_visual = np.clip(
    diff,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(DIFF_IMG),
    diff_visual
)


# ============================================================
# UART TIMING
# ============================================================

elapsed = end_time - start_time

total_uart_bytes = (
    len(input_pixels)
    +
    len(hardware)
)

effective_rate = (
    total_uart_bytes / elapsed
    if elapsed > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SOFTWARE VS FPGA RESULTS")
print("=" * 65)

print("\nFULL 64x64 IMAGE")

print(
    f"Max pixel difference : {max_diff}"
)

print(
    f"Mean pixel difference: {mean_diff:.6f}"
)

print(
    f"MAE                  : {mae:.6f}"
)

print(
    f"RMSE                 : {rmse:.6f}"
)

print(
    f"Exact matching pixels: "
    f"{exact_match}/{NUM_PIXELS}"
)

print(
    f"Exact match rate     : "
    f"{100 * exact_match / NUM_PIXELS:.2f}%"
)


print("\nINTERIOR 62x62")

print(
    f"Max pixel difference : {inner_max}"
)

print(
    f"Mean pixel difference: {inner_mean:.6f}"
)

print(
    f"Exact matching pixels: "
    f"{inner_exact}/{inner_total}"
)

print(
    f"Exact match rate     : "
    f"{100 * inner_exact / inner_total:.2f}%"
)


print("\nUART")

print(
    f"Round-trip time: "
    f"{elapsed * 1000:.3f} ms"
)

print(
    f"Transferred bytes: "
    f"{total_uart_bytes}"
)

print(
    f"Effective byte rate: "
    f"{effective_rate:.2f} bytes/s"
)


# ============================================================
# SAVE REPORT
# ============================================================

report = f"""
FPGA SOFTWARE VERIFICATION REPORT
=================================

Image size:
{WIDTH} x {HEIGHT}

Pixels:
{NUM_PIXELS}

UART:
{PORT}
{BAUD} baud
8-N-1

FULL IMAGE
----------
Max difference:
{max_diff}

Mean difference:
{mean_diff:.6f}

MAE:
{mae:.6f}

RMSE:
{rmse:.6f}

Exact matches:
{exact_match}/{NUM_PIXELS}

Exact match rate:
{100 * exact_match / NUM_PIXELS:.4f} %

INTERIOR 62x62
--------------
Max difference:
{inner_max}

Mean difference:
{inner_mean:.6f}

Exact matches:
{inner_exact}/{inner_total}

Exact match rate:
{100 * inner_exact / inner_total:.4f} %

UART ROUND TRIP
---------------
{elapsed * 1000:.3f} ms
"""

REPORT_FILE.write_text(
    report,
    encoding="utf-8"
)


print("\nSaved:")
print(HARDWARE_BIN)
print(HARDWARE_IMG)
print(DIFF_IMG)
print(REPORT_FILE)

print("\nFPGA VERIFICATION COMPLETE")