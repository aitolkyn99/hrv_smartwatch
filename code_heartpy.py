import scipy
import pandas as pd
import numpy as np
import heartpy as hp

import os
import csv
import time
import statsmodels.api as sm
from datetime import datetime, timedelta
from scipy.signal import resample

######################################################################################################################
win_size = 60000 * 5
min_bpm_hz = 0.8 
max_bpm_hz = 3.67 
new_fs_threshold = 10 
######################################################################################################################

HR_VAL = 'HR'

def clean_slices(df):
    seconds_counter = 0
    start_date_second = df['fullTime'].iloc[0].split(':')[-1]

    for index, row in df.iterrows():
        current_date_second = row['fullTime'].split(':')[-1]

        if current_date_second == start_date_second:
            seconds_counter += 1
        else:
            if seconds_counter < 10: # each slice should have 10 datapoints
                return index
            
            seconds_counter = 0
            
def filter_daily(df):
    df.reset_index(drop=True, inplace=True)
    idx_to_start = clean_slices(df)
    df = df[idx_to_start::]

    return df

### Detect watch - OFF wrist signal
def aggregate_hrm_seconds(df):
    # convert hr column to float:
    df[HR_VAL] = df[HR_VAL].astype(float)

    dd1 = df.groupby('fullTime').agg({'ts': 'min', HR_VAL:['mean','median']})
    dd1.columns = ['ts','HR_mean','HR_median']
    dd1=dd1.reset_index()

    dd2 = (df[HR_VAL]<=10).groupby(df['fullTime']).sum().reset_index(name='num_off_wrist')

    hrm_aggregated = pd.merge(dd1, dd2, on='fullTime', how='left')
    
    print("== ground-truth labeling process finished! ==")
    return hrm_aggregated[hrm_aggregated['num_off_wrist'] == 0]

def do_preprocessing(df_ppg_raw, df_hrm_raw):
    ## pre-processing the ppg signal by referring to the Samsung HR data (consider it as a ground-truth): ##
    df_ppg = filter_daily(df_ppg_raw)
    df_hrm = filter_daily(df_hrm_raw)
    hrm_aggregated = aggregate_hrm_seconds(df_hrm)

    merged_ppg_hrm_filtered = pd.merge(df_ppg, hrm_aggregated, on='fullTime').rename(columns={'ts_x': 'ts'})
    merged_ppg_hrm_filtered.reset_index(inplace=True, drop=True)
    return merged_ppg_hrm_filtered




def ppg_preProcessing_chunk(df_daily, win_size):
    i = 0; j = 0; chunk_ = []
    while i >= 0:
        try:
            chunk_.append(j)

            # in case a gap exists, and it is greater than 'win_size'
            if int(df_daily['ts'][i]) < int(df_daily['ts'][i+1]) - win_size:
                j += 1

            i += 1

        except KeyError:
            break

    df_daily['chunk'] = chunk_


def ppg_preProcessing_slicing(df_daily, win_size, dict_slice_vol):
    i = 0; j = 0; k = 0; index_ = []; slice_ = []
    start_time = df_daily['ts'][i]
    end_time = start_time + win_size

    while i >= 0:
        try:
            index_.append(i)
            slice_.append(k)
            if df_daily.chunk[i] == df_daily.chunk[i+1]: 
                if df_daily['ts'][i] > end_time:
                    dict_slice_vol[k] = i-j
                    start_time = df_daily['ts'][i]
                    end_time = start_time + win_size
                    j = i; k += 1
                    
            else:
                if (i+1-j < 2):
                    print("[NaN] data-points # per slice: " + str(i+1-j))
                else:
                    print("Preprocessing_slicing:", str(df_daily.chunk[i]) + ',' + str(k) + ',' + str(i+1-j))

                dict_slice_vol[k] = i+1-j
                start_time = df_daily['ts'][i+1]
                end_time = start_time + win_size
                j = i+1; k += 1

            i += 1

        except KeyError:
            dict_slice_vol[k] = i+1-j
            print("in except", str(df_daily.chunk[i]) + ',' + str(k) + ',' + str(i+1-j))
            break

    df_daily['slice'] = slice_
    df_daily['index'] = index_
    
    return dict_slice_vol

def extract_sampling_rate(timer):
    ## keep the time part from date ##
    timer = [x.split(' ')[-1] for x in timer]

    # extract the sampling rate 
    fs = hp.get_samplerate_datetime(timer, timeformat='%H:%M:%S.%f')

    return fs


def isolate_HR_frequencies(signal, fs, min_bpm_hz, max_bpm_hz):              
    ## Apply a standard butterworth bandpass filter to remove everything < min_bpm_hz and > max_bpm_hz: ##
    filtered_ = hp.filter_signal(signal, [min_bpm_hz, max_bpm_hz], sample_rate=fs, order=3, filtertype='bandpass')

    return filtered_


def resample_signal(filtered_signal, fs, fs_range):
    resampled = resample(filtered_signal, len(filtered_signal) * fs_range)

    # compute the new sampling rate
    new_fs = fs * fs_range
    return (resampled, new_fs)


def detect_RR(resampled, new_fs):
    wd, m = hp.process(resampled, sample_rate = new_fs, calc_freq=True, high_precision=True, high_precision_fs=1000.0)
    return (wd, m)


def do_feature_extraction(original_df, min_bpm_hz, max_bpm_hz, new_fs_threshold, ppg_slice_no):
    slice_label = original_df.tail(1).slice.values[0]    
    print("slice label", slice_label)
    df_hrv = pd.DataFrame()
    counter = 0
    original_df['fullTime_ms'] = [(datetime.utcfromtimestamp(i / 1000).replace(microsecond=i % 1000 * 1000)+timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S.%f") for i in original_df['ts']]
        
    for i in range(0,int(slice_label)+1):
        slice_ = original_df.loc[original_df.slice == i]
        slice_from = slice_['index'].head(1).values[0]
        slice_to = slice_['index'].tail(1).values[0]
        print("========")
        print(slice_from)
        print(slice_to)
        print(ppg_slice_no[i])
        print("--------")
        
        signal_slice = original_df['ppg'].values[slice_from:slice_to]
        timer_slice = original_df['fullTime_ms'].values[slice_from:slice_to]
        ts_start = original_df['ts'].values[slice_from:slice_to]
        
        if ppg_slice_no[i] > 20:
            # Step 1: Extract sampling rate of the sample
            fs = extract_sampling_rate(timer_slice)
            print("init fs: ", fs)

            '''
            According to the Nyquist–Shannon theorem, when sampling a continuous signal the sampling rate should be at least twice that of 
            the highest frequency to be captured ==> 3.67*2.
            '''
            if fs > 7.34: 
                try:
                    # Step 2: Isolate HR frequencies
                    hr_signal = isolate_HR_frequencies(signal_slice, fs, min_bpm_hz, max_bpm_hz)
                    # Step 3: Upsample signal to higher frequencies - interpolate more points (helps the peak detection)
                    resampled, new_fs = resample_signal(hr_signal, fs, new_fs_threshold)
                    print('new fs: ', new_fs)


                    print(resampled[0:int(new_fs * ppg_slice_no[i])].shape)
                    wd, hrv_analysis = detect_RR(resampled[0:int(new_fs * ppg_slice_no[i])], new_fs)
                    print('=====================DETECTED RR========================')
                    observed_ibi = len(wd['RR_list_cor'])

                    mean_hr = original_df['HR_mean'].values[slice_from:slice_to].mean()
                    missingess_ppg = 1 - ((observed_ibi+1)/float(hrv_analysis['bpm']*5))
                    missingess_samsung_bpm = 1 - ((observed_ibi+1)/float(mean_hr*5))                
                    hr_diff = abs(hrv_analysis['bpm'] - mean_hr)

                    print(slice_from, slice_to)
                    print('ppg bpm:', hrv_analysis['bpm'], ' missingess_ppg: ', missingess_ppg)
                    print('samsung bpm:', mean_hr, ' missingess_samsung_bpm: ', missingess_samsung_bpm)
                    print('hr_diff: ', hr_diff)
                    print("========")

                    df_hrv.at[counter, 'ts_start'] = ts_start[0]
                    df_hrv.at[counter, 'fulltime_start'] = timer_slice[0]
                    df_hrv.at[counter, 'fulltime_end'] = timer_slice[-1]
                    df_hrv.at[counter, 'missingess_ppg'] = missingess_ppg
                    df_hrv.at[counter, 'missingess_samsung_bpm'] = missingess_samsung_bpm
                    df_hrv.at[counter, 'bpm'] = hrv_analysis['bpm']
                    df_hrv.at[counter, 'mean_hr_samsung'] = mean_hr
                    df_hrv.at[counter, 'hr_diff'] = hr_diff
                    df_hrv.at[counter, 'ibi'] = hrv_analysis['ibi']
                    df_hrv.at[counter, 'sdnn'] = hrv_analysis['sdnn']
                    df_hrv.at[counter, 'sdsd'] = hrv_analysis['sdsd']
                    df_hrv.at[counter, 'rmssd'] = hrv_analysis['rmssd']
                    df_hrv.at[counter, 'pnn20'] = hrv_analysis['pnn20']
                    df_hrv.at[counter, 'pnn50'] = hrv_analysis['pnn50']
                    df_hrv.at[counter, 'hr_mad'] = hrv_analysis['hr_mad']
                    df_hrv.at[counter, 'breathingrate'] = hrv_analysis['breathingrate']
                    df_hrv.at[counter, 'lf'] = hrv_analysis['lf']
                    df_hrv.at[counter, 'hf'] = hrv_analysis['hf']
                    df_hrv.at[counter, 'lf/hf'] = hrv_analysis['lf/hf']
                    df_hrv.at[counter, 'observed_ibi'] = observed_ibi

                    counter += 1

                except Exception as e:
                    print('error', e, timer_slice[0], timer_slice[-1])

            else:
                print("Nyquist criterion violated:", timer_slice[0], timer_slice[-1], fs)
            
        else:
            print("[Exception] data-points # per slice: " + str(ppg_slice_no[i]))

    return df_hrv

def do_extractingHRV(df_ppg_filtered):
    print(df_ppg_filtered.head())
    dict_slice_vol = {}
    if len(df_ppg_filtered) > 0:
        ppg_preProcessing_chunk(df_ppg_filtered, win_size)
        ppg_slice_no = ppg_preProcessing_slicing(df_ppg_filtered, win_size, dict_slice_vol)

        ## extracting the HRV features from the pre-processed ppg signal: ##
        extracted_features = do_feature_extraction(df_ppg_filtered, min_bpm_hz, max_bpm_hz, new_fs_threshold, ppg_slice_no)
        return extracted_features



'''
rootdir containing 49 directories of 49 participants with the following files: 
1) all_days_ppg.csv.gz for PPG signals collected from the user 
2) all_days_hrm.csv.gz for HR collected from the user 
'''

rootdir = 'data_per_user/'
SAVE_PATH = '' # specify the path where to save the output of this code

for user in os.listdir(rootdir):
    if '.' in user: continue
    try:
        c = pd.read_csv(rootdir+user+'/all_days_ppg.csv.gz').sort_values('ts').reset_index(drop=True)
        c['fullTime'] = [time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(int(i/1000))) for i in c.ts]

        c2 = pd.read_csv(rootdir+user+'/all_days_hrm.csv.gz').sort_values('ts').reset_index(drop=True)
        c2['fullTime'] = [time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(int(i/1000))) for i in c2.ts]
        
        data1 = do_preprocessing(c,c2)
        
        in_data = data1[(data1.ppg>=0)&(data1.ppg<=4194304)].reset_index(drop=True)
        data2 = do_extractingHRV(in_data)
        data2.to_csv(SAVE_PATH+'/hrv_computed_'+user+'.csv.gz', index=False)
        print("Completed for user:", user)
    except:
        print("error on user:", user)