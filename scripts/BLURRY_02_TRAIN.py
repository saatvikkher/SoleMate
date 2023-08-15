from process_util import process_image_blurry
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./BLURRY_02_TRAIN.log', 'w')

print("[BLURRY 02 TRAIN] Started.")

# Read in training split
km_train = pd.read_csv("Jul24_submission/pair_info/BLURRY_02_TRAIN_KM.csv")
knm_train = pd.read_csv("Jul24_submission/pair_info/BLURRY_02_TRAIN_KNM.csv")

combined_train = pd.concat([km_train, knm_train], ignore_index=True)

# Grab the file names and mated status of Q and K
Q_files = combined_train.q.values
K_files = combined_train.k.values
mated = combined_train.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_train)):
    start = time.time()
    if i % 20 == 0:
        print("[BLURRY 02 TRAIN] Progress: ", (i*100) / len(combined_train))
    try:
        row = pd.DataFrame(process_image_blurry(
            Q_files[i], K_files[i], mated[i]), index=[0])
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv("RESULT_BLURRY_02_TRAIN.csv", index=False)
    except Exception as e:
        print("[BLURRY 02 TRAIN] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[BLURRY 02 TRAIN]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[BLURRY 02 TRAIN] Complete!")
