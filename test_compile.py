"""
Test script to verify torch.compile behavior with CROWPEAS_COMPILE environment variable.
"""
import os
import sys
import torch

# Set the environment variable to prevent compilation by default
os.environ["CROWPEAS_COMPILE"] = "0"

# Create a simple model to test
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CROWPEAS_COMPILE: {os.environ.get('CROWPEAS_COMPILE', '0')}")

# This is the relevant part of the fix we added to core.py
print("\nTest without compilation:")
try:
    # Without setting the environment variable to true, compilation should be skipped
    if os.environ.get("CROWPEAS_COMPILE", "0").lower() in ("1", "true", "yes"):
        print("Attempting to compile the model...")
        compiled_model = torch.compile(model)
        print("Model successfully compiled with torch.compile")
    else:
        print("Skipping compilation (CROWPEAS_COMPILE not enabled)")
except Exception as e:
    print(f"Test without compilation failed: {e}")

# Now test with compilation enabled
print("\nTest with compilation enabled:")
os.environ["CROWPEAS_COMPILE"] = "1"
try:
    if os.environ.get("CROWPEAS_COMPILE", "0").lower() in ("1", "true", "yes"):
        print("Attempting to compile the model...")
        compiled_model = torch.compile(model)
        print("Model successfully compiled with torch.compile")
    else:
        print("Skipping compilation (CROWPEAS_COMPILE not enabled)")
except Exception as e:
    print(f"Compilation failed (expected if no C++ compiler is available): {e}")
    print("The code is designed to gracefully handle this case.")