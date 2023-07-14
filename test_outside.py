from process_util import process_image
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./test_outside.log', 'w')

print("[TEST OUTSIDE] Started.")

# Read in testing split
km_test = pd.read_csv("KM_test.csv")
knm_test = pd.read_csv("KNM_test.csv")

combined_test = pd.concat([km_test, knm_test], ignore_index=True)

# Grab the file names and mated status of Q and K
Q_files = combined_test.q.values
K_files = combined_test.k.values
mated = combined_test.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_test)):
    start = time.time()
    if i % 20 == 0:
        print("[TEST OUTSIDE] Progress: ", (i*100) / len(combined_test))
    try:
        row = pd.DataFrame(process_image(
            Q_files[i], K_files[i], mated[i], partial_type="outside"), index=[0])
        df = pd.concat([df, row], ignore_index=True)    
        df.to_csv("result_test_outside_0707.csv", index=False)
    except Exception as e:
        print("[TEST OUTSIDE] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[TEST OUTSIDE]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[TEST OUTSIDE] Complete!")
