@echo off

echo "Started BASELINE_TEST_0_250"
start /B python -m Jul17_submission.BASELINE_TEST_0_250

echo "Started BASELINE_TEST_250_END"
start /B python -m Jul17_submission.BASELINE_TEST_250_END

echo "Started BASELINE_TRAIN_0_200"
start /B python -m Jul17_submission.BASELINE_TRAIN_0_200

echo "Started BASELINE_TRAIN_200_400"
start /B python -m Jul17_submission.BASELINE_TRAIN_200_400

echo "Started BASELINE_TRAIN_400_600"
start /B python -m Jul17_submission.BASELINE_TRAIN_400_600

echo "Started BASELINE_TRAIN_600_800"
start /B python -m Jul17_submission.BASELINE_TRAIN_600_800

echo "Started BASELINE_TRAIN_800_1000"
start /B python -m Jul17_submission.BASELINE_TRAIN_800_1000

echo "Started BASELINE_TRAIN_1000_END"
start /B python -m Jul17_submission.BASELINE_TRAIN_1000_END

echo "Started DIFF_TIME_1VS2_TEST_300_END"
start /B python -m Jul17_submission.DIFF_TIME_1VS2_TEST_300_END

echo "Started DIFF_TIME_1VS3_TRAIN_0_250"
start /B python -m Jul17_submission.DIFF_TIME_1VS3_TRAIN_0_250

echo "Started DIFF_TIME_1VS3_TRAIN_250_END"
start /B python -m Jul17_submission.DIFF_TIME_1VS3_TRAIN_250_END

echo "Started INSIDE_TEST"
start /B python -m Jul17_submission.INSIDE_TEST

echo "Started OUTSIDE_TEST"
start /B python -m Jul17_submission.OUTSIDE_TEST

echo "Started TOE_TEST_KNM"
start /B python -m Jul17_submission.TOE_TEST_KNM

echo "Submitted all jobs"
