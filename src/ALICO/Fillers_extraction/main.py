"""
# TODO: write description
"""


from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import textgrids
import os
# type of data comes from extract_label_timesteps_from_file() function. It is
# Dict[label->timesteps]. Label is a string value, can be something like 'mhm' and so on.
from scipy.io import wavfile

labels_timesteps_data_type=Dict[str,List[Tuple[float, float]
                                        ]
                                ]

def extract_label_timesteps_from_file(path:str, labels:Tuple[str,...])->labels_timesteps_data_type:
    # TODO: write description
    try:
        grid = textgrids.TextGrid(path)
    except Exception:
        a=1+2
    result_intervals={}
    for label in labels:
        result_intervals[label]=[]
    for item in grid['ORT']:
        label=str(item.text)
        if label in labels:
            result_intervals[label].append((item.xmin, item.xmax))

    return result_intervals

def extract_label_timesteps_from_files_in_dir(path_to_dir:str, labels:Tuple[str,...])->List[Tuple[str, labels_timesteps_data_type]]:
    # TODO: write description
    # figure out, what files directory contains
    filenames=os.listdir(path_to_dir)
    # check if some files have not extention TextGrid
    for filename in filenames:
        if filename.split('.')[-1]!='TextGrid':
            raise ValueError('One of the files in provided directory have different from TextGrid extension. Filename: %s' % filename)
    # iterate through files
    extracted_labels_timesteps = []
    for filename in filenames:
        label_timesteps=extract_label_timesteps_from_file(path=os.path.join(path_to_dir, filename),
                                                          labels=labels)
        extracted_labels_timesteps.append((filename,label_timesteps))
    return extracted_labels_timesteps

def read_wav_file(path:str)->Tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    return sample_rate, data

def write_wav_file(path:str, data:np.ndarray, sample_rate:int)->None:
    wavfile.write(path, sample_rate, data)


def extract_utterance_according_timesteps(path_to_file:str, timesteps:Tuple[float, float], additional_interval:float=0):
    # TODO: write description
    sample_rate, wav_file = read_wav_file(path_to_file)
    # recalculate start and end point in terms of indexes (int). Expand window by additional_interval as well.
    start_in_int=int(np.round(timesteps[0]*sample_rate)-np.round(additional_interval*sample_rate))
    end_in_int=int(np.round(timesteps[1]*sample_rate)+np.round(additional_interval*sample_rate))
    # check if we are out of range of wav_file
    if start_in_int<0:
        start_in_int = 0
        end_in_int = int(timesteps[1]*sample_rate)
    if end_in_int>wav_file.shape[0]:
        end_in_int =wav_file.shape[0]
        start_in_int = wav_file.shape[0]-int(timesteps[0]*sample_rate)
    # calculate relative timings for fillers in seconds
    relative_start_filler=int(timesteps[0]*sample_rate-start_in_int)
    relative_end_filler=relative_start_filler+int((timesteps[1]-timesteps[0])*sample_rate)
    return wav_file[start_in_int: end_in_int], sample_rate, (relative_start_filler, relative_end_filler)

def extract_utterances_from_all_files(filename_timesteps:List[Tuple[str, labels_timesteps_data_type]], additional_interval:float,
                                      path_to_data:str, output_path:str)->None:
    # create output directory, if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # create dataframe with all meta information for every extracted filler
    meta_information=pd.DataFrame(columns=['relative_path','filler_start_idx','filler_end_idx', 'filler_type'])
    # iterate through all instances
    for filename, label_timesteps in filename_timesteps:
        labels=list(label_timesteps.keys())
        filename_without_extention=filename.split('.')[0]
        for label in labels:
            # create dir with name of audiofile and inside it with names of labels
            if not os.path.exists(os.path.join(output_path, filename_without_extention,label)):
                os.makedirs(os.path.join(output_path, filename_without_extention,label), exist_ok=True)
            # iterate over all timesteps with assigned to this concrete label
            for start_point, end_point in label_timesteps[label]:
                # extract utterance from audio
                if start_point is not None and end_point is not None:
                    absolut_path=os.path.join(output_path, filename_without_extention,label,
                                               '%.2f_%.2f.wav'%(start_point, end_point))
                    relative_path=os.path.join(filename_without_extention,label,
                                               '%.2f_%.2f.wav'%(start_point, end_point))
                    # extract utterance
                    audio_utterance, sample_rate, \
                    relative_filler_timings=extract_utterance_according_timesteps(path_to_file=os.path.join(path_to_data, filename_without_extention+'.wav'),
                                                                          timesteps=(start_point, end_point),
                                                                          additional_interval=additional_interval)
                    # save all meta information
                    new_row={'relative_path':relative_path,
                             'filler_start_idx':relative_filler_timings[0],
                             'filler_end_idx':relative_filler_timings[1],
                             'filler_type':label
                    }
                    meta_information=meta_information.append(new_row, ignore_index=True)
                    # save extracted utterance with interval
                    write_wav_file(path=absolut_path, data=audio_utterance, sample_rate=sample_rate)
    meta_information.to_csv(os.path.join(output_path, 'metainformation.csv'), index=False)


def extract_utterance_with_deleted_filler_according_timesteps(path_to_file:str, timesteps:Tuple[float, float], additional_interval:float=0):
    # TODO: write description
    sample_rate, wav_file = read_wav_file(path_to_file)
    # recalculate start and end point in terms of indexes (int). Expand window by additional_interval as well.
    start_in_int=int(np.round(timesteps[0]*sample_rate)-np.round(additional_interval*sample_rate))
    end_in_int=int(np.round(timesteps[1]*sample_rate)+np.round(additional_interval*sample_rate))
    # check if we are out of range of wav_file
    if start_in_int<0:
        start_in_int = 0
        end_in_int = int(timesteps[1]*sample_rate)
    if end_in_int>wav_file.shape[0]:
        end_in_int =wav_file.shape[0]
        start_in_int = wav_file.shape[0]-int(timesteps[0]*sample_rate)
    # calculate relative timings for fillers in seconds
    relative_start_filler=int(timesteps[0]*sample_rate-start_in_int)
    relative_end_filler=relative_start_filler+int((timesteps[1]-timesteps[0])*sample_rate)
    return wav_file[start_in_int: end_in_int, 0], sample_rate, (relative_start_filler, relative_end_filler)

def extract_utterances_with_deleted_filler_from_all_files(filename_timesteps:List[Tuple[str, labels_timesteps_data_type]], additional_interval:float,
                                      path_to_data:str, output_path:str)->None:
    # create output directory, if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # create dataframe with all meta information for every extracted filler
    meta_information=pd.DataFrame(columns=['relative_path','filler_start_idx','filler_end_idx', 'filler_type'])
    # iterate through all instances
    for filename, label_timesteps in filename_timesteps:
        labels=list(label_timesteps.keys())
        filename_without_extention=filename.split('.')[0]
        for label in labels:
            # create dir with name of audiofile and inside it with names of labels
            if not os.path.exists(os.path.join(output_path, filename_without_extention,label)):
                os.makedirs(os.path.join(output_path, filename_without_extention,label), exist_ok=True)
            # iterate over all timesteps with assigned to this concrete label
            for start_point, end_point in label_timesteps[label]:
                # extract utterance from audio
                if start_point is not None and end_point is not None:
                    absolut_path=os.path.join(output_path, filename_without_extention,label,
                                               '%.2f_%.2f.wav'%(start_point, end_point))
                    relative_path=os.path.join(filename_without_extention,label,
                                               '%.2f_%.2f.wav'%(start_point, end_point))
                    # extract utterance
                    audio_utterance, sample_rate, \
                    relative_filler_timings=extract_utterance_with_deleted_filler_according_timesteps(path_to_file=os.path.join(path_to_data, filename_without_extention+'.wav'),
                                                                          timesteps=(start_point, end_point),
                                                                          additional_interval=additional_interval)
                    # save all meta information
                    new_row={'relative_path':relative_path,
                             'filler_start_idx':relative_filler_timings[0],
                             'filler_end_idx':relative_filler_timings[1],
                             'filler_type':label
                    }
                    meta_information=meta_information.append(new_row, ignore_index=True)
                    # save extracted utterance with interval
                    write_wav_file(path=absolut_path, data=audio_utterance, sample_rate=sample_rate)
    meta_information.to_csv(os.path.join(output_path, 'metainformation.csv'), index=False)


if __name__ == '__main__':
    path_to_labels=r'E:\Databases\ALICO\ALICO\alico_coop\segmentation'
    path_to_data=r'E:\Databases\ALICO\ALICO\recordings'
    for add_interval in (1.5,):
        path_to_output=r'E:\Databases\ALICO\ALICO\extracted_utterances_with_deleted_fillers_%.1f'%add_interval
        #grid = textgrids.TextGrid(path_to_file)
        label_timesteps=extract_label_timesteps_from_files_in_dir(path_to_labels, labels=('ja','m','mhm','okay','achso','ah'))
        extract_utterances_with_deleted_filler_from_all_files(label_timesteps, additional_interval=add_interval,
                                          path_to_data=path_to_data, output_path=path_to_output)
