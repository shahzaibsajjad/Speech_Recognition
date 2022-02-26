import xmltodict
import numpy as np
import pandas as pd
import os


from scipy.io import wavfile

path_to_xml_files= 'C:\\Users\\Shah Zaib Sajjad\\Desktop\\Speech Recog\\MapTask_Ger\\l1_exmaralda_2.1\\l1_exmaralda_2.1'
path_to_wav_files= 'C:\\Users\\Shah Zaib Sajjad\\Desktop\\Speech Recog\\MapTask_Ger\\l1_wav_2.1\\l1_wav_2.1'
path_to_save= 'C:\\Users\\Shah Zaib Sajjad\\Desktop\\Speech Recog\\MapTask_Ger\\extracted_fillers_code_Test_New_data'

xml_filelist = sorted(os.listdir(path_to_xml_files))
wav_filelist = sorted(os.listdir(path_to_wav_files))

#params
list_of_filler_types=['f1', 'f2', 'f3', 'ff1']
entities=['instructor_df','instructee_df']
add_sec_to_segment=1.5
path_to_save+='_'+str(add_sec_to_segment)

if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)

print(xml_filelist)
print(wav_filelist)

for file_idx in range(len(xml_filelist)):
    sample_rate, wav_file = wavfile.read(os.path.join(path_to_wav_files,wav_filelist[file_idx]))
    with open(os.path.join(path_to_xml_files, xml_filelist[file_idx]), encoding='utf-8') as fd:
        xml_file = xmltodict.parse(fd.read(), encoding='utf-8')
    # create directory to save fillers in this concrete wav file
    directory_to_save=os.path.join(path_to_save, wav_filelist[file_idx].split('.')[0])
    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
    # extracting timesteps for every event
    event_timing={}
    tmp_timing=xml_file['basic-transcription']['basic-body']['common-timeline']['tli']
    for event in tmp_timing:
        event_id=event['@id']
        event_time=float(event['@time'])
        event_timing[event_id]=event_time

    # tiers, which are annotations of every word, discorse particles, ...
    extracted_utterances=[]
    tiers=xml_file['basic-transcription']['basic-body']['tier']
    for tier in tiers:
        if tier['@category'] in entities and 'event' in tier:

            events=tier['event']
            if not isinstance(events, list):
                events=[events]
            for event in events:
                if '#text' in event and event['#text'] in list_of_filler_types:
                    event_start_id=event['@start']
                    event_end_id=event['@end']
                    # calculate start and end indexes to cut wav_file
                    event_start_idx=int(np.round(event_timing[event_start_id]*sample_rate-add_sec_to_segment*sample_rate))
                    event_end_idx=int(np.round(event_timing[event_end_id]*sample_rate+add_sec_to_segment*sample_rate))
                    # cut wav_file
                    # if event_start_id<0 or event_end_id> the length of wav, then pad with zeros
                    if event_start_idx<0:
                        audio_chunk=wav_file[0:event_end_idx]
                    else:
                        audio_chunk = wav_file[event_start_idx:event_end_idx]
                    overall_length=event_end_idx-event_start_idx
                    if event_start_idx<0:
                        zero_pad=np.zeros((-event_start_idx, wav_file.shape[1]))
                        audio_chunk=np.concatenate([zero_pad, audio_chunk], axis=0)
                    if event_end_idx>wav_file.shape[0]:
                        zero_pad=np.zeros((event_end_idx-wav_file.shape[0],wav_file.shape[1]))
                        audio_chunk=np.concatenate([audio_chunk, zero_pad], axis=0)
                    assert(audio_chunk.shape[0]==overall_length)
                    extracted_audio=audio_chunk
                    label=event['#text']
                    extracted_utterances.append([extracted_audio, label, event_start_id, event_end_id])
    # save all extracted audios according their labels
    # create directories for each filler_category
    for filler_type in list_of_filler_types:
        if not os.path.exists(os.path.join(directory_to_save, filler_type)):
            os.mkdir(os.path.join(directory_to_save,filler_type))
    # save extracted audios
    for extracted_audio, label, event_start_id, event_end_id in extracted_utterances:
        filename='%s_%s_%s.wav'%(label, event_start_id, event_end_id)
        full_path=os.path.join(directory_to_save,label, filename)
        wavfile.write(full_path, rate=sample_rate, data=extracted_audio)
