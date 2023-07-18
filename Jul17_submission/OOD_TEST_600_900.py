from process_util import process_image_OOD
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./OOD_TEST_600_900.log', 'w')

print("[TEST OOD] Started.")

# Read in training split
km_test = pd.read_csv("Jul17_submission/pair_info/OOD_TEST_KM.csv")
knm_test = pd.read_csv("Jul17_submission/pair_info/OOD_TEST_KNM.csv")
combined_test = pd.concat([km_test, knm_test], ignore_index=True)[600:900]

# Grab the file names and mated status of Q and K
Q_files = combined_test.q.values
K_files = combined_test.k.values
mated = combined_test.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_test)):
    start = time.time()
    if i % 20 == 0:
        print("[TEST OOD] Progress: ", (i*100) / len(combined_test))
    try:
        row = pd.DataFrame(process_image_OOD(
            Q_files[i], K_files[i], mated[i], partial_type="full", 
            folder_path="OOD/"), index=[0])
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv("RESULT_OOD_TEST_600_900.csv", index=False)
    except Exception as e:
        print("[TEST OOD] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[TEST OOD]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[TEST OOD] Complete!")