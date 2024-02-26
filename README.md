# In-situ wearable device-based dataset of continuous HRV monitoring accompanied by sleep diaries

- Goal 1: to preprocess the data and then to extract the HRV featuers by going through "heartpy" library

- Function:
1) do_preprocessing(): to preprocess the PPG signals by referring Samsung HR as a ground-truth (i.e., going through aggregate_hrm_seconds()) => discard any PPG signals if any abnormal values from the corresponding ground-truth are obsered at lease one time per second (c.f., in general, there will be 10 PPG signals in a second, meaning 10 hz)
2) do_extractingHRV(): to label slice based on "win_size" (i.e., going through ppg_preProcessing_chunk() then ppg_preProcessing_slicing()), and then to extract HRV features (i.e., going through do_feature_extraction()) => the extracted HRV features are collated as one .csv file per the chunk label
3) do_mergeHRV(): to collate all HRV features .csv files of the chunks to make one .csv file of the entire dates

- Hyperparameter:
1) chunk_no = # of chunks that you want to split on the entire PPG dataset of a user ** it should be same to that of "00_Extracting_dataset.py" **
2) win_size = the size of a slice (millisecond level), a slice should be set for a granularity to go through "heartpy" library, currenntly it is set for 5 minutes (60000 * 5)
3) min_bpm_hz & max_bpm_hz: min & max of the reasonable range of heart rates (hz level)
4) new_fs_threshold: a value that can multiply to the current sampling ratio in order to upsample (conventionally, it is good to be at least 100hz for HRV analysis)
5) user: a specific deviceId that you want to analyze






- Goal 2: to postprocess of the derived HRV features, and to extract reported labels independent to the heart signals
  
- Function:
1) discard_slices_by_missingness(): to discard any slices if their "missingness_samsung_bpm" values are greater than the hyperparameter "missingess_score"
2) extract_sd_per_hour(): to group slices on an hour level and to compute mean, median, std, max-min per hour for each HRV feature
3) compute_residual(): to compute residual of the IBI feature per hour
4) compute_average_hrv(): to compute the average values of the median, std, max-min, median at the night time (2am - 5am) for each day

- Hyperparameter:
1) missingess_score: a threshold of the missingess_score between 0 to 1, a greater value means less rigorous, if a N-min slice ls greater than missingess_score, that slice is discarded
2) min_required_points: a threshold of the required data-ponints, a greater value means more rigorous, if a 1-hr window ls less than min_required_points, that window is discarded
3) user: a specific deviceId that you want to analyze
