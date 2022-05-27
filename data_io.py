import os
import numpy as np
from tqdm import tqdm
import pickle

vna_header = "# Header:freq_Hz,real,imag"

def write_vna_data(output_filename,freqs,z,meta_data = None):
    with open(output_filename,'w') as f:
        #if metadata is not None:
        #    f.write(F"{metadata}\n")
        for f_value, real, imag in list(zip(freqs, np.real(z),np.imag(z))):
            f.write(F"{f_value},{real},{imag}\n")

def read_vna_data(input_filename):
    freqs = []
    real = []
    imag = []
    with open(input_filename, "r") as f:
        raw_lines = [raw_line.strip() for raw_line in f.readlines()]
    for line_index, raw_line in tqdm(list(enumerate(raw_lines))):
        if raw_lines[line_index][0] == "#":
            pass
        else:
            split = raw_lines[line_index].split(",")
            freqs.append(float(split[0]))
            real.append(float(split[1]))
            imag.append(float(split[2]))
            
    return np.asarray(freqs),np.asarray(real+1j*np.asarray(imag))
            
def write_iq_sweep_data(output_filename,freqs,z,meta_data = None):
    iq_dict = {"freqs":freqs,"z":z}
    file_to_write = open(output_filename, "wb")
    pickle.dump(iq_dict, file_to_write)

def read_iq_sweep_data(input_filename):
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
    return data['freqs'],data['z']

def write_stream_data(output_filename,freqs,z,meta_data = None):
    stream_dict = {"freqs":freqs,"z":z}
    file_to_write = open(output_filename, "wb")
    pickle.dump(stream_dict, file_to_write)

def read_stream_data(input_filename):
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
    return data['freqs'],data['z']

def write_filename_set(output_filename,filenames):
    with open(output_filename,'w') as f:
        for filename in filenames:
            f.write(filename+"\n")

def read_filename_set(input_filename):
    filenames = []
    with open(input_filename,'w') as f:
        raw_lines = [raw_line.strip() for raw_line in f.readlines()]
        for line in raw_lines:
            filenames.append(line)
    return filenames
            