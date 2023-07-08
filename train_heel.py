from process_util import process_image
import pandas as pd
import sys
import gc
import time

# Redirect the standard output to the log file
sys.stdout = open('./train_heel.log', 'w')

print("[TRAIN HEEL] Started.")

# Read in training split
km_train = pd.read_csv("KM_train.csv")
knm_train = pd.read_csv("KNM_train.csv")

combined_train = pd.concat([km_train, knm_train], ignore_index=True)

# Grab the file names and mated status of Q and K
Q_files = combined_train.q.values
K_files = combined_train.k.values
mated = combined_train.mated.values  # Value of 'mated' parameter

df = pd.DataFrame()

for i in range(len(combined_train)):
    start = time.time()
    if i % 20 == 0:
        print("[TRAIN HEEL] Progress: ", (i*100) / len(combined_train))
    try:
        row = pd.DataFrame(process_image(
            Q_files[i], K_files[i], mated[i], partial_type="heel"), index=[0])
        df = pd.concat([df, row], ignore_index=True)    
        df.to_csv("result_train_heel_0707.csv", index=False)
    except Exception as e:
        print("[TRAIN HEEL] Caught error at index " + str(i) + str(e))
    end = time.time()
    print("[TRAIN HEEL]: Time for iteration ", str(i), ": ", str(end - start))
    gc.collect()

print("[TRAIN HEEL] Complete!")
