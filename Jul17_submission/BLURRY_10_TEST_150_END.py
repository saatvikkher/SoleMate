from process_util import process_image_blurry
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./BLURRY_10_TEST_150_END.log', 'w')

print("[BLURRY 10 TEST] Started.")

# Read in training split
km_train = pd.read_csv("Jul17_submission/pair_info/BLURRY_10_TEST_KM.csv")
knm_train = pd.read_csv("Jul17_submission/pair_info/BLURRY_10_TEST_KNM.csv")

combined_train = pd.concat([km_train, knm_train], ignore_index=True)[150:]

# Grab the file names and mated status of Q and K
Q_files = combined_train.q.values
K_files = combined_train.k.values
mated = combined_train.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_train)):
    start = time.time()
    if i % 20 == 0:
        print("[BLURRY 10 TEST] Progress: ", (i*100) / len(combined_train))
    try:
        row = pd.DataFrame(process_image_blurry(
            Q_files[i], K_files[i], mated[i]), index=[0])
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv("RESULT_BLURRY_10_TEST_150_END.csv", index=False)
    except Exception as e:
        print("[BLURRY 10 TEST] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[BLURRY 10 TEST]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[BLURRY 10 TEST] Complete!")
