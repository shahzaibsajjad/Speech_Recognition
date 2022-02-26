import os

import numpy as np
import pandas as pd

from src.BeMaTaC.Fillers_extraction.Fillers_extraction import fillers_extractor_BeMaTaC

if __name__ == "__main__":
    # params
    path_to_xml_files = r'C:\Users\Shah Zaib Sajjad\Desktop\Speech Recog\MapTask_Ger\l1_exmaralda_2.1\l1_exmaralda_2.1'
    path_to_wav_files = r'C:\Users\Shah Zaib Sajjad\Desktop\Speech Recog\MapTask_Ger\l1_wav_2.1\l1_wav_2.1'
    path_to_save = r'C:\Users\Shah Zaib Sajjad\Desktop\Speech Recog\MapTask_Ger\extracted_fillers_main_run'
    list_of_filler_types = ['f1', 'f2', 'f3', 'ff1']
    entities = ['instructor_df', 'instructee_df']
    pad_sec_to_segment = 1.5
    cut_off_filler=True
    path_to_save += '_' + str(pad_sec_to_segment)

    xml_filelist = sorted(os.listdir(path_to_xml_files))
    wav_filelist = sorted(os.listdir(path_to_wav_files))

    # create dir if does not exists
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    for file_idx in range(len(xml_filelist)):
        extractor=fillers_extractor_BeMaTaC(path_to_wavfile=os.path.join(path_to_wav_files,wav_filelist[file_idx]),
                                            path_to_xml_file=os.path.join(path_to_xml_files, xml_filelist[file_idx]))
        # create directory to save fillers in this concrete wav file
        directory_to_save_fillers_from_file = os.path.join(path_to_save, wav_filelist[file_idx].split('.')[0])
        if not os.path.exists(directory_to_save_fillers_from_file):
            os.mkdir(directory_to_save_fillers_from_file)
        # extract event timings
        event_timings=extractor.extract_event_timings_from_xml_dict(extractor.xml_file)
        # extract utterances with defined pad lengths
        extracted_utterances=extractor.extract_utterances(xml_dict=extractor.xml_file,
                                                          wav_file=extractor.wav_file,
                                                          event_timings=event_timings,
                                                          entities=entities,
                                                          list_of_filler_types=list_of_filler_types,
                                                          pad_sec=pad_sec_to_segment)
        # save all extracted audios according their labels
        # create directories for each filler_category
        for filler_type in list_of_filler_types:
            if not os.path.exists(os.path.join(directory_to_save_fillers_from_file, filler_type)):
                os.mkdir(os.path.join(directory_to_save_fillers_from_file, filler_type))
        # save extracted audios
        extractor.save_utterances_in_dir(path_to_dir=directory_to_save_fillers_from_file,
                                         extracted_utterances=extracted_utterances, cut_off_filler=cut_off_filler)