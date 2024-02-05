import os
import numpy as np
from tqdm import tqdm
import pickle
from dataclasses import fields
import resonator_class as resonator_class

vna_header = "# Header:freq_Hz,real,imag"

def write_vna_data(output_filename,freqs,z,meta_data = None,save_as_pickle = False):
    if save_as_pickle:
        dict = {'freqs':freqs,'z':z,'meta_data':meta_data}
        with open(output_filename+".p", 'wb') as f:
            pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(output_filename+".csv",'w') as f:
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
    for line_index, raw_line in tqdm(list(enumerate(raw_lines)),ascii = True):
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

def write_stream_data(output_filename,freqs,z,ttl_data = None,meta_data = None, downsampling = None): # Downsampling added by AnV on 01/19/2024
    stream_dict = {"freqs":freqs,"z":z,"ttl":None,"downsampling":None}
    if ttl_data is not None:
        stream_dict['ttl'] = ttl_data
    if downsampling is not None:
        stream_dict['downsampling'] = downsampling;
    file_to_write = open(output_filename, "wb")
    print(output_filename); 
    pickle.dump(stream_dict, file_to_write)

def read_stream_data(input_filename): # Downsampling option added by AnV on 01/19/2024
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
        # print(data['downsampling']); just a test
    if 'downsampling' in data.keys():
        if 'ttl' in data.keys():
            return data['freqs'],data['z'],data['ttl'],data['downsampling']
        return data['freqs'],data['z'],data['downsampling']
    else:
        if 'ttl' in data.keys():
            return data['freqs'],data['z'],data['ttl']
        else:
            return data['freqs'],data['z']


def write_filename_set(output_filename,filenames,value_list_1 = None):
    with open(output_filename,'w') as f:
        for k,filename in enumerate(filenames):
            if value_list_1 is not None:
                f.write(filename+","+str(value_list_1[k])+"\n")
            else:
                f.write(filename+"\n")

def read_filename_set(input_filename):
    filenames = []
    value_list_1 = []
    with open(input_filename,'r') as f:
        raw_lines = [raw_line.strip() for raw_line in f.readlines()]
        for line in raw_lines:
            if len(line.split(","))>1:
                filenames.append(line.split(",")[0])
                value_list_1.append(line.split(",")[1])
            else:
                filenames.append(line)
    if len(value_list_1)>0:
        return filenames,value_list_1
    else:
        return filenames
            

def write_resonators_class(output_filename,res_class):
    with open(output_filename,'w') as f:
        # write header
        line = "#"
        for field in fields(res_class.resonators[0]):
            line += field.name+","
        line = line[:-1] # strip last comma
        f.write(line+"\n")
        for resonator in res_class.resonators:
            line = ""
            for field in fields(resonator):
                field_str = str(getattr(resonator,field.name))
                if "," in field_str: #handle list of flags
                    field_str = field_str.replace(",","|")
                    field_str = field_str.replace(" ","")
                line += field_str+","
            line = line[:-1] # strip last comma
            f.write(line+"\n")

def read_resonators_class(input_filename):
    res_class = resonator_class.resonators_class([])
    with open(input_filename,'r') as f:
        print("hello")
        raw_lines = [raw_line.strip() for raw_line in f.readlines()]
        print(raw_lines)
        for line in raw_lines:
            print(line)
            if line[0] == "#": #header:
                field_names = []
                split = line[1:].split(",")
                for split in line[1:].split(","):
                    field_names.append(split)
            else:
                split = line.split(",")
                res_class.resonators.append(resonator_class.resonator(split[0]))# frequency required
                for k,field in enumerate(fields(res_class.resonators[-1])):
                    if 'float' in str(field.type):
                        setattr(res_class.resonators[-1],field.name,float(split[k])) 
                    elif 'bool' in str(field.type):
                        if split[k] == "True":
                            setattr(res_class.resonators[-1],field.name,True)
                        else:
                            setattr(res_class.resonators[-1],field.name,False)
                    elif 'List' in str(field.type):
                        list_str = split[k].replace("|",",")
                        #print(list_str)
                        list_of_strings = []
                        if "," in list_str:
                            for string in list_str[1:-1].split(","):# strip brackets
                                list_of_strings.append(string[1:-1])# strip '
                                #print(list_of_strings)
                        setattr(res_class.resonators[-1],field.name,list_of_strings)
                    else:
                        print("type "+str(field.type)+ " not recognized")
    return res_class
