import numpy as np
import time
import sys

j = 1439941
rows = 3240
Y_f_rows = 3000

print("Generating random matrices (double precision)...")
t0 = time.time()
top = np.random.randn(rows, j)
Y_f = np.random.randn(Y_f_rows, j)
print(f"Generation took: {time.time() - t0:.2f}s")
print(f"Memory used by top: {top.nbytes / 1e9:.2f} GB")
print(f"Memory used by Y_f: {Y_f.nbytes / 1e9:.2f} GB")

# Test QR method
print("\n--- Running QR solver ---")
t0 = time.time()
Q, R = np.linalg.qr(top.T, mode='reduced')
print(f"QR decomposition took: {time.time() - t0:.2f}s")
print(f"Q shape: {Q.shape}, R shape: {R.shape}")

t0_mul = time.time()
Y_f_Q = Y_f @ Q
print(f"Matrix multiplication (Y_f @ Q) took: {time.time() - t0_mul:.2f}s")

t0_solve = time.time()
L = np.linalg.solve(R.T, Y_f_Q.T).T
print(f"Solve (R.T @ L.T = Y_f_Q.T) took: {time.time() - t0_solve:.2f}s")
print(f"Total QR method time: {time.time() - t0:.2f}s")
print(f"L shape: {L.shape}")

# Optional: Test pinv method but only on a smaller scale or with warnings
print("\n--- Running pinv solver (Benchmarking) ---")
print("Warning: this will allocate 37 GB for pinv(top).")
sys.stdout.flush()
t0 = time.time()
try:
    top_pinv = np.linalg.pinv(top)
    print(f"pinv(top) took: {time.time() - t0:.2f}s")
    print(f"top_pinv shape: {top_pinv.shape}")
    t0_mul = time.time()
    L_pinv = Y_f @ top_pinv
    print(f"Matrix multiplication (Y_f @ top_pinv) took: {time.time() - t0_mul:.2f}s")
    print(f"Total pinv method time: {time.time() - t0:.2f}s")
    print(f"L_pinv shape: {L_pinv.shape}")
    print(f"Max error between QR and pinv: {np.max(np.abs(L - L_pinv)):.2e}")
except Exception as e:
    print(f"pinv failed: {e}")
