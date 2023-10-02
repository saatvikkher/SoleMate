from process_util import process_image
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./TOE_TEST.log', 'w')

print("[TOE TEST] Started.")

# Read in testing split
km_test = pd.read_csv("Jul24_submission/pair_info/BASELINE_TEST_KM.csv")
knm_test = pd.read_csv("Jul24_submission/pair_info/BASELINE_TEST_KNM.csv")

combined_test = pd.concat([km_test, knm_test], ignore_index=True)

# Grab the file names and mated status of Q and K
Q_files = combined_test.q.values
K_files = combined_test.k.values
mated = combined_test.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_test)):
    start = time.time()
    if i % 20 == 0:
        print("[TOE TEST] Progress: ", (i*100) / len(combined_test))
    try:
        row = pd.DataFrame(process_image(
            Q_files[i], K_files[i], mated[i], partial_type="toe"), index=[0])
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv("RESULT_TOE_TEST.csv", index=False)
    except Exception as e:
        print("[TOE TEST] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[TOE TEST]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[TOE TEST] Complete!")