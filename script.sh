#!/bin/bash

echo "Started BASELINE_TEST_0_250"
python -m Jul17_submission.BASELINE_TEST_0_250 &
pid1 = $!
echo "Started BASELINE_TEST_250_END"
python -m Jul17_submission.BASELINE_TEST_250_END &
pid2 = $!

echo "Started BASELINE_TRAIN_0_200"
python -m Jul17_submission.BASELINE_TRAIN_0_200 &
pid3 = $!
echo "Started BASELINE_TRAIN_200_400"
python -m Jul17_submission.BASELINE_TRAIN_200_400 &
pid4 = $!
echo "Started BASELINE_TRAIN_400_600"
python -m Jul17_submission.BASELINE_TRAIN_400_600 &
pid5 = $!
echo "Started BASELINE_TRAIN_600_800"
python -m Jul17_submission.BASELINE_TRAIN_600_800 &
pid6 = $!
echo "Started BASELINE_TRAIN_800_1000"
python -m Jul17_submission.BASELINE_TRAIN_800_1000 &
pid7 = $!
echo "Started BASELINE_TRAIN_1000_END"
python -m Jul17_submission.BASELINE_TRAIN_1000_END &
pid8 = $!

echo "Started DIFF_TIME_1VS2_TEST_300_END"
python -m Jul17_submission.DIFF_TIME_1VS2_TEST_300_END &
pid9 = $!

echo "Started DIFF_TIME_1VS3_TRAIN_0_250"
python -m Jul17_submission.DIFF_TIME_1VS3_TRAIN_0_250 &
pid10 = $!
echo "Started DIFF_TIME_1VS3_TRAIN_250_END"
python -m Jul17_submission.DIFF_TIME_1VS3_TRAIN_250_END &
pid11 = $!

echo "Started INSIDE_TEST"
python -m Jul17_submission.INSIDE_TEST &
pid12 = $!
echo "Started OUTSIDE_TEST"
python -m Jul17_submission.OUTSIDE_TEST &
pid13 = $!
echo "Started TOE_TEST_KNM"
python -m Jul17_submission.TOE_TEST_KNM &
pid14 = $!

echo "Submitted all jobs"
