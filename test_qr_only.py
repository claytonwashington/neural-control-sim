import numpy as np
import time

j = 1439941
rows = 3240
Y_f_rows = 3000

print("Generating random matrices (double precision)...", flush=True)
t0 = time.time()
top = np.random.randn(rows, j)
Y_f = np.random.randn(Y_f_rows, j)
print(f"Generation took: {time.time() - t0:.2f}s", flush=True)

print("\n--- Running QR solver ---", flush=True)
t0 = time.time()
Q, R = np.linalg.qr(top.T, mode='reduced')
print(f"QR decomposition took: {time.time() - t0:.2f}s", flush=True)
print(f"Q shape: {Q.shape}, R shape: {R.shape}", flush=True)

t0_mul = time.time()
Y_f_Q = Y_f @ Q
print(f"Matrix multiplication (Y_f @ Q) took: {time.time() - t0_mul:.2f}s", flush=True)

t0_solve = time.time()
L = np.linalg.solve(R.T, Y_f_Q.T).T
print(f"Solve (R.T @ L.T = Y_f_Q.T) took: {time.time() - t0_solve:.2f}s", flush=True)
print(f"Total QR method time: {time.time() - t0:.2f}s", flush=True)
print("Done.", flush=True)
