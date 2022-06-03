import matplotlib.pyplot as plt
import numpy as np
import qsghw.core.helpers as corehelpers
import qsghw.fpgaip.helpers as fpgaiphelpers
import qsghw.misc.windows as windows
import qsghw.core.packetparser as packetparser
import time
from instruments import hp83732a as hp
from instruments import weinchel8310 as weinchel
import resonator_class as res_class
from KIDs import find_resonances_interactive as find_res
from KIDs import calibrate as cal
from KIDs import resonance_fitting as res_fit
import tune_resonators as tune_resonators
import os
import data_io as data_io
import glob
from tqdm import tqdm
from scipy import signal
yeses = ['yes','y']

class readout(object):

    def __init__(self):
        self.ctrl = "tcp://192.168.0.128"
        self.window_type = "roach2"
        self.merge = 256
        self.decimate = 1
        self.amp = 0.4
        self.output = "FINEDDC"
        self.data = "Auto"
        self.open_data = True
        self.quiet = True
        self.insane = False
        self.regaccess = False
        self.show_access = True
        self.debug = False
        self.verbose = False
        self.group = 0
        self.fshift = -24
        self.pshift = 0
        self.qshift = -16
        self.pktsize = 8192
        self.tag = None
        self.frequencies = [-19.875*10**6,-50*10**6]
        self.amplitude = 0.4
        self.lo = hp.synthesizer()
        self.lo_freq = 500.00*10**6
        self.lo.set_frequency(self.lo_freq)
        self.data = "udp://10.0.0.10"
        self.iq_sweep_data = None
        self.input_attenuator = weinchel.attenuator() #input to detectors, output of fpga
        self.output_attenuator = weinchel.attenuator('GPIB0::8::INSTR')
        self.res_class = res_class.resonators_class([])
        self.vna_data = None
        self.tau = 66*10**-9
        self.stream_data = None
        self.data_dir = "~/multitone_data"
        self.data_dir = os.path.expanduser(self.data_dir)
        print(self.data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.vna_save_filename = "vna_sweep.csv"
        self.iq_sweep_save_filename = "iq_sweep.p"
        self.stream_save_filename = "stream.p"
        self.res_class_save_filename = "resonators.csv"
        self.iq_sweep_iq_fits = None
        self.power_sweep_iq_data = None
        self.power_sweep_iq_fits = None
        self.normalizing_amplitudes = None
        
        (self.ctrlif, self.dataif, self.eids) = corehelpers.open(self.ctrl, self.data, opendata=self.open_data,
                                                                 verbose=self.debug, quiet=self.quiet)

        self.ctrlif.debug = [self.insane, self.show_access]

        self.wfplayer = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":wfplayer:")
        self.frontend = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":chpfbfft:")
        self.circb = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":circbuffer:", "")
        self.wfplayer.debug = self.frontend.debug = self.frontend.pfbfft.debug = self.frontend.binsel[self.group].debug \
            = self.frontend.fineddc[self.group].debug = self.circb.debug = (self.regaccess, self.debug, self.verbose)
        assert self.wfplayer and self.frontend and self.circb

        self.samplerate = self.wfplayer.samplerate = self.frontend.get_samplerate()
        self.iqorder = self.wfplayer.iqorder = True if self.frontend.get_iqorder() else False
        print()
        print("samplerate = %8.3f MHz   iqorder = %d" % (self.samplerate/1e6, self.iqorder))
        print()

        # stop flow of data
        self.frontend.write_monstop(1)

        # generate and write window - accept exteneded window names
        # okay to overwirte window?
        self.window = windows.mkwinbyextname(self.window_type, self.frontend.get_windowlen(), self.frontend.get_fftlen())
        print("window =", self.window_type, len(self.window))

        self.frontend.write_fshift(self.fshift)
        self.frontend.write_pshift(self.pshift)
        self.frontend.write_qshift(self.qshift)
        print("bit shifts after filter pfbfft fineddc =", self.frontend.read_fshift(), self.frontend.read_pshift(),
              self.frontend.read_qshift())
        print()

        # frontend output selection
        print("frontend output = %s   merge = %3d   decimate = %3d" % (self.output, self.merge, self.decimate))
        self.frontend.change_output(self.output, self.merge, self.decimate)

        # apply tag
        if self.tag: self.frontend.write_tag(int(self.tag, 0))

        # not frontend, but anyways: program and reset circbuffer
        self.circb.write_packetsize(self.pktsize)
        self.circb.write_bufspace(0) # 0 => maximum supported
        self.circb.reset()
        print("circbuf packetsize = %5d   bufspace = %6d" % (self.circb.read_packetsize(), self.circb.read_bufspace()))

        self.set_frequencies(self.frequencies,self.amp)

        # reset every frontend block, restart flow
        self.frontend.write_rstall(1)
        self.frontend.write_rstall(0)
        self.frontend.write_monstop(0)

        # check that data is flowing: print conters 4 times
        for i in range(4):
            f = self.frontend # shorthand
            print("running = %d   counters = %8d %8d %8d %8d" % (f.read_monrunning(), f.read_monvalcnt(), f.read_monlastcnt(),
                                                                 f.read_monpktcnt(), f.read_monlosscnt()))
            time.sleep(0.05)
        print()

        print("wfplayer freq res = %10.6f kHz" % (self.wfplayer.get_tone_freq_res() / 1000))
        print("frontend freq res = %10.6f kHz" % (self.frontend.get_chan_freq_res() / 1000))
        print()

    def variables(self):
        print("readout object's variables")
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],np.ndarray):
                print(key,"numpy array with shape ",self.__dict__[key].shape)
            elif isinstance(self.__dict__[key],list):
                print(key,"list of length",len(self.__dict__[key]))
            elif key == 'res_class':
                print(key,"Resonator class with len ",len(self.__dict__[key].resonators))
            else:
                print(key,self.__dict__[key])

    def set_frequencies(self,frequencies,amplitude,random_phase = True,fill_DAQ_range = True):
        '''
        set frequencies
        a lot of this should probably be migrated into the wfplayer class
        '''
        try:
            len(frequencies)
        except:
            frequencies = [frequencies]
        if np.isscalar(amplitude): # supplying array or of amplitudes?
            amplitude = np.ones(len(frequencies))*amplitude
        if random_phase:
            np.random.seed()
            phase = np.random.uniform(0., 360, len(frequencies))
        else:
            phase = np.zeros(len(frequencies))
        
        
        n_tones = len(frequencies)
        self.merge = 512//n_tones #how many samples in a packet
        print("merge = "+str(self.merge))
        self.frontend.change_output(self.output, self.merge, self.decimate)
        self.real_freq_list = []
        self.wfplayer.del_tone(groupid = self.group)
        for i in range(0,len(frequencies)):
            self.wfplayer.set_tone(self.group,i,frequencies[i],phase[i],phase[i],amplitude[i],amplitude[i])
            self.real_freq_list.append(self.wfplayer.get_tone(self.group,i)[0])

        self.wfplayer.write_tone_list()
        
        if fill_DAQ_range: # this is ineffecient currently writing tones to board twice
            maximum_of_wave = np.max((np.abs(np.real(self.wfplayer.wave)),np.abs(np.imag(self.wfplayer.wave))))
            scale_factor = (0.8*(2**16//2-1))/maximum_of_wave 
            amplitude = amplitude*scale_factor
            self.real_freq_list = []
            self.wfplayer.del_tone(groupid = self.group)
            for i in range(0,len(frequencies)):
                self.wfplayer.set_tone(self.group,i,frequencies[i],phase[i],phase[i],amplitude[i],amplitude[i])
                self.real_freq_list.append(self.wfplayer.get_tone(self.group,i)[0])
            self.wfplayer.write_tone_list()
            
        # old way with list 
        #self.real_freq_list = self.wfplayer.set_tone_list(self.group, frequencies, write=True, amp=amplitude)
        self.chanlist = self.frontend.set_chan_list(self.group, self.real_freq_list, write=True)
        self.binnolist = [ self.frontend.get_chan(self.group, i)[1] for i in self.chanlist ]

        print()
        print("     asked         actual       fftbin  channel")
        for i in range(len(frequencies)):
            print("  %8.3f MHz   %8.3f MHz   %6d   %6d" % (frequencies[i]/1e6, self.real_freq_list[i]/1e6,
                                                           self.binnolist[i], self.chanlist[i]))

        print()


    def iq_sweep(self,span = 100*10**3,npts = 101,average = 1024*10,offset = 0.0,plot = True):
        to_read = int(1024*average) # at some point should turn this into an integration time
        self.iq_sweep_freqs_lo = np.linspace(self.lo_freq-span/2+offset,self.lo_freq+span/2+offset,npts)
        self.iq_sweep_freqs = np.zeros((npts,len(self.real_freq_list)))
        for m in range(0,len(self.real_freq_list)):
            self.iq_sweep_freqs[:,m] = self.iq_sweep_freqs_lo+self.real_freq_list[m]
        self.iq_sweep_data = np.zeros((len(self.iq_sweep_freqs),len(self.real_freq_list)),dtype ='complex')
        for n, freq in tqdm(enumerate(self.iq_sweep_freqs_lo),total = npts):
            self.lo.set_frequency(freq)
            time.sleep(0.1)
            self.lo.get_frequency()
            alldata_array = self.get_data(to_read,verbose = False)
            for m in range(0,alldata_array.shape[1]):
                self.iq_sweep_data[n,m] = np.mean(alldata_array[:,m])
        if plot:
            self.plot_iq_sweep()

    def plot_iq_sweep(self):
        if self.iq_sweep_data is not None:
            ip = tune_resonators.interactive_plot(self.iq_sweep_freqs,self.iq_sweep_data,retune = False,
                                                  combined_data = self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,:]/10**6,
                                                  combined_data_names = ["Resonator Frequencies (MHz)"])        
        else:
            print("No IQ sweep data found: unable to plot")

    def retune_resonators(self,find_min = False,write_resonator_tones = True):
        if self.res_class.len_use_true() != self.iq_sweep_data.shape[1]:
            print("number of resonators in iq data does not match number of currently"+\
                  "active resonator cannot retune because iq information is ambiguouse")
            return
            # could allow this if I do some matching to figure out what is what
        #    confirmation = input("number of resonators in iq data does not match"+\
        #                         "number of currently active resonators: Proceed? y/n")
        #    if confirmation in yeses: #should I allow this
        #        pass 
        #    else:
        #        return
        new_frequencies = tune_resonators.tune_kids(self.iq_sweep_freqs,self.iq_sweep_data,find_min = find_min)
        if self.res_class.len_use_true() == self.iq_sweep_data.shape[1]:
            # change freqeuncies in resonator class
            new_frequencies_index = 0
            for resonator in self.res_class.resonators:
                if resonator.use:
                    resonator.frequency = new_frequencies[new_frequencies_index]
                    new_frequencies_index += 1
        if write_resonator_tones:
            self.write_resonator_tones()
            
            

    def vna_sweep(self,span,resolution = 1*10**3,average = 1024*10):
        '''
        Pretend we are vna for finding resonances
        currently limited to putting tone ~10MHz apart
        '''
        n_tones = span//(10*10**6)
        if n_tones >100:
            n_tones = 100
        print("Using "+str(n_tones)+" for VNA sweep")
        self.frequencies = list(np.linspace(-span/2,span/2,n_tones))
        tone_spacing = self.frequencies[1]-self.frequencies[0]
        npts = int(tone_spacing//resolution)
        
        iq_sweep_freqs = np.linspace(self.lo_freq-tone_spacing/2,self.lo_freq+tone_spacing/2,npts)        

        self.set_frequencies(self.frequencies,amplitude = 0.8/n_tones)
        
        self.vna_freqs = np.asarray(())
        for m in range(0,len(self.real_freq_list)):
            self.vna_freqs = np.append(self.vna_freqs,self.real_freq_list[m]+iq_sweep_freqs)

        to_read = int(1024*average) # at some point should turn this into an integration time                                           
        iq_sweep_data = np.zeros((len(iq_sweep_freqs),len(self.real_freq_list)),dtype ='complex')
        for n, freq in enumerate(iq_sweep_freqs):
            self.lo.set_frequency(freq)
            time.sleep(0.01)
            curr_freq = self.lo.get_frequency()
            print(str(curr_freq/10**6)+" MHz")
            alldata_array = self.get_data(to_read,verbose = False)
            for m in range(0,alldata_array.shape[1]):
                iq_sweep_data[n,m] = np.mean(alldata_array[:,m])

        self.vna_data = np.asarray(()) 
        for m in range(0,len(self.real_freq_list)):
            self.vna_data = np.append(self.vna_data,iq_sweep_data[:,m])



    def get_stream_data(self,stream_len= 1024*1024*1024):
        self.lo.set_frequency(self.lo_freq) #make sure LO is back at nominal frequency
        time.sleep(0.1)
        self.stream_data = self.get_data(stream_len,verbose = False)
        self.stream_frequencies = [freq + self.lo_freq for freq in self.real_freq_list]

    def plot_iq_and_stream(self,decimate = 1000):#eventualy decimate -> frequency
        # decimate only handels 1 sig fig i.e 2000 no 2500
        if self.stream_data is not None:
            if self.iq_sweep_data is not None:
                self.decimated_stream_data = self.stream_data #need deep copy?
                factors_of_10 = int(np.floor(np.log10(decimate)))
                for k in range(0,factors_of_10):
                    self.decimated_stream_data = signal.decimate(self.decimated_stream_data,10,axis = 0)
                self.decimated_stream_data = signal.decimate(self.decimated_stream_data,
                                                             decimate//(10**factors_of_10),axis =0)
                    
                ip = tune_resonators.interactive_plot(self.iq_sweep_freqs,self.iq_sweep_data,
                                                      stream_data = self.decimated_stream_data,retune = False)
            else:
                print("No IQ sweep data found: unable to plot")
        else:
            print("No stream data found: unable to plot")
        
        
    def plot_vna_sweep(self):
        plt.figure(1)
        plt.plot(self.vna_freqs/10**6,20*np.log10(np.abs(self.vna_data)))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power dB")
        plt.title("VNA Sweep")
        plt.show()

    def save_vna_data(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.vna_save_filename
        data_io.write_vna_data(save_filename,self.vna_freqs,self.vna_data)

    def load_vna_data(self,filename = None):
        if not filename: # look for most recent file
            list_of_files = glob.glob(self.data_dir+'/*'+self.vna_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.vna_freqs,self.vna_data = data_io.read_vna_data(latest_file)
        else:
            try:
                self.vna_freqs,self.vna_data = data_io.read_vna_data(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)

    def save_iq_data(self,filename_suffix = ""):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.iq_sweep_save_filename.split(".")[0]+\
            filename_suffix+self.iq_sweep_save_filename.split(".")[1]
        data_io.write_iq_sweep_data(save_filename,self.iq_sweep_freqs,self.iq_sweep_data)
        return save_filename
        
    def load_iq_data(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.iq_sweep_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.iq_sweep_freqs,self.iq_sweep_data = data_io.read_iq_sweep_data(latest_file)
        else:
            try:
                self.iq_sweep_freqs,self.iq_sweep_data = data_io.read_iq_sweep_data(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)

    def save_stream_data(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.stream_save_filename
        data_io.write_stream_data(save_filename,self.stream_frequencies,self.stream_data)
        return save_filename

    def load_stream_data(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.stream_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.stream_freqs,self.stream_data = data_io.read_stream_data(latest_file)
        else:
            try:
                self.stream_frequencies,self.stream_data = data_io.read_stream_data(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)
                    
    def find_kids(self):
        if list(self.vna_data): #pythonic booleaness of list/None does not work on np.arrays() 
            if len(self.res_class.resonators)==0: #first time finding resonators
                ip = find_res.find_vna_sweep(self.vna_freqs,self.vna_data)
                for m in range(len(ip.kid_idx)):
                    self.res_class.resonators.append(res_class.resonator(ip.chan_freqs[ip.kid_idx[m]],
                                                                         use = True))
                    self.res_class.resonators[m].flags = ip.flags[m]
            else:
                print("please write code to handle retunning from vna")
                        
        else:
            print("No VNA data found")

    def save_res_class(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.res_class_save_filename
        data_io.write_resonators_class(save_filename,self.res_class)

    def load_res_class(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.res_class_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.res_class = data_io.read_resonators_class(latest_file)
        else:
            try:
                self.res_class = data_io.read_resonators_class(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)
        
            

    def get_data(self,to_read,verbose= True):
        self.dataif.reset()
        (nbytes,buffer) = self.dataif.readall(to_read)
        (headeroffsets, payloadoffsets, payloadlengths, seqnos) = packetparser.findlongestcontinuous(buffer, incr=1024)
        (consumed, alldata) = packetparser.parsemany(buffer, headeroffsets[0], payloadoffsets, verbose = verbose)
        alldata_array = np.zeros(alldata.data.shape,dtype = 'complex')
        alldata_array = alldata.data['i']+1j*alldata.data['q']
        return alldata_array


    def close(self):
        self.dataif.close()
        self.ctrlif.close()


    def get_ADC_wave(self):
        # frontend output selection                                                                                             
        #print("frontend output = %s   merge = %3d   decimate = %3d" % (self.output, self.merge, self.decimate))
        self.frontend.write_monstop(1)
        self.frontend.change_output("INPUT", 1, 8193)
        # not frontend, but anyways: program and reset circbuffer                                                               
        self.circb.write_packetsize(self.pktsize)
        self.circb.write_bufspace(0) # 0 => maximum supported                                                                   
        self.circb.reset()
        print("circbuf packetsize = %5d   bufspace = %6d" % (self.circb.read_packetsize(), self.circb.read_bufspace()))
        # reset every frontend block, restart flow                                                                              
        self.frontend.write_rstall(1)
        self.frontend.write_rstall(0)
        self.frontend.write_monstop(0)

        time.sleep(1)
        self.adc_wave = self.get_data(32*1024)
        self.adc_wave = self.adc_wave[0,:]
        # change back to normal operation
        self.frontend.write_monstop(1)
        self.frontend.change_output(self.output, self.merge, self.decimate)
        # not frontend, but anyways: program and reset circbuffer
        self.circb.write_packetsize(self.pktsize)
        self.circb.write_bufspace(0) # 0 => maximum supported                                                                   
        self.circb.reset()
        print("circbuf packetsize = %5d   bufspace = %6d" % (self.circb.read_packetsize(), self.circb.read_bufspace()))
        # reset every frontend block, restart flow
        self.frontend.write_rstall(1)
        self.frontend.write_rstall(0)
        self.frontend.write_monstop(0)


    def write_resonator_tones(self,amplitude = None):
        frequencies = []
        for m in range(len(self.res_class.resonators)):
            if self.res_class.resonators[m].use == True:
                frequencies.append(self.res_class.resonators[m].frequency-self.lo_freq)

        if amplitude is not None:
            if np.sum(amplitude)>0.8:
                amplitude = amplitude/np.sum(amplitude)*0.8
            print("Normalizing tone powers")
            self.set_frequencies(frequencies,amplitude)
        elif self.normalizing_amplitudes is not None:
            amplitude = self.normalizing_amplitudes
            if np.sum(amplitude)>0.8:
                amplitude = amplitude/np.sum(amplitude)*0.8
            print("Normalizing tone powers")
            self.set_frequencies(frequencies,amplitude)
        else:
            self.set_frequencies(frequencies,0.8/len(frequencies))

    def mask_by_separation(self,required_separation):
        '''
        turn off tones that are too close together
        '''
        last_active_frequency = None
        for resonator in sorted(self.res_class.resonators):
            if not last_active_frequency:#handle first res
                print("First resonator")
                last_active_frequency = sorted(self.res_class.resonators)[0].frequency
                resonator.use = True
            elif resonator.frequency<last_active_frequency+required_separation:
                print("Turning resonator off")
                resonator.use = False
            else:
                print("Leaving resonator on")
                resonator.use = True
                last_active_frequency = resonator.frequency

    def take_noise_set(self,gain_span = 1e6,fine_span = 100e3,stream_len = 1024*1024*1024,plot = True):
        filenames = []
        print("taking gain sweep")
        self.iq_sweep(gain_span,plot = False)
        filenames.append(self.save_iq_data())
        print("taking fine sweep")
        self.iq_sweep(fine_span,plot = False)
        filenames.append(self.save_iq_data())
        print("taking streaming data")
        self.get_stream_data(stream_len)
        filenames.append(self.save_stream_data())
        self.save_noise_set_filenames(filenames)
        if plot:
            self.plot_iq_and_stream()
        

    def save_noise_set_filenames(self,filenames):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_noise_set_filenames.txt"
        data_io.write_filename_set(save_filename,filenames)
        

    def fit_cable_delay(self,plot = True):
        if list(self.vna_data):
            phase = np.arctan2(np.real(self.vna_data),np.imag(self.vna_data))
            self.tau,phase_gradient = cal.fit_cable_delay_from_slope(self.vna_freqs,phase,plot = plot)
            print("tau = "+str(self.tau))
        else:
            print("No VNA data found")


    def power_sweep(self,max_attenuation,min_attenuation,attenuation_step,span=100*10**3,npts = 101,
                    average = 1024*10,output_attenuation = None,plot = False):
        '''
        do iq sweeps at different power levels to find bifurcation power levels
        output_attenutaion should be set to a single value that is good is at 
        halfway between max and min attenuation
        '''
        n_attn_levels = np.int((max_attenuation-min_attenuation)/attenuation_step)+1
        attn_levels = np.linspace(max_attenuation,min_attenuation,n_attn_levels)
        n_res = len(self.chanlist)
        self.power_sweep_iq_data = np.zeros((npts,n_res,n_attn_levels),dtype = 'complex')
        self.power_sweep_iq_freqs = np.zeros((npts,n_res,n_attn_levels)) 
        if output_attenuation is not None:
            if self.output_attenuator is not None:
                attn_levels_output = attn_levels[::-1]-(attn_levels[len(attn_levels)//2]-output_attenuation)
            else:
                print("No output attenuator connected do not specify output_attenuation")
                return
        else:
            if self.output_attenuator is not None: # best guess
                output_attenuation = self.output_attenuator.get_attenuation()
                attn_levels_output = attn_levels[::-1]-(attn_levels[len(attn_levels)//2]-output_attenuation)
        if self.output_attenuator is not None:
            index_lt_zero = np.where(attn_levels_output<0)
            attn_levels_output[index_lt_zero] = 0
            
        filenames = []
        for k in tqdm(range(0,n_attn_levels)):
            self.input_attenuator.set_attenuation(attn_levels[k])
            input_attenuation = self.input_attenuator.get_attenuation()
            if self.output_attenuator is not None:
                self.output_attenuator.set_attenuation(attn_levels_output[k])
                output_attenuation = self.output_attenuator.get_attenuation()
                print("Input Attenuation = "+str(input_attenuation)+" Output_attenaution = " +\
                      str(output_attenuation))
            else:
                print("Input Attenuation = "+str(input_attenuation))
            self.iq_sweep(span,npts,average,plot = False)
            self.power_sweep_iq_data[:,:,k] = self.iq_sweep_data
            self.power_sweep_iq_freqs[:,:,k] = self.iq_sweep_freqs
            filenames.append(self.save_iq_data())
        self.attn_levels = attn_levels
        self.save_power_sweep_filenames(filenames,attn_levels)
        
        
    def save_power_sweep_filenames(self,filenames,attn_levels):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_power_sweep_filenames.txt"
        data_io.write_filename_set(save_filename,filenames,attn_levels)

    def load_power_sweep_data(self,input_filename_of_filenames = None):
        if not input_filename_of_filenames:
            list_of_files = glob.glob(self.data_dir+'/*'+"_power_sweep_filenames.txt")
            latest_file = max(list_of_files, key=os.path.getctime)
            filenames,self.attn_levels  = data_io.read_filename_set(latest_file)
            print(self.attn_levels)
            print(filenames)
        else:
            try:
                filenames,self.attn_levels = data_io.read_filename_set(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                    return
                else:
                    print("error with file "+filename)
                    return
        self.attn_levels = np.asarray(self.attn_levels,dtype = 'float')
        n_attn_levels = len(filenames)
        self.load_iq_data(filenames[0])
        self.power_sweep_iq_data = np.zeros((self.iq_sweep_data.shape[0],self.iq_sweep_data.shape[1],n_attn_levels),dtype = 'complex')
        self.power_sweep_iq_freqs = np.zeros((self.iq_sweep_data.shape[0],self.iq_sweep_data.shape[1],n_attn_levels))
        self.power_sweep_iq_data[:,:,0] = self.iq_sweep_data
        self.power_sweep_iq_freqs[:,:,0] = self.iq_sweep_freqs
        for k in range(1,n_attn_levels):
            self.load_iq_data(filenames[k])
            self.power_sweep_iq_data[:,:,k] = self.iq_sweep_data
            self.power_sweep_iq_freqs[:,:,k] = self.iq_sweep_freqs
            
    def fit_iq_sweep(self):
        if self.iq_sweep_data is not None:
            self.iq_sweep_iq_fits = res_fit.fit_nonlinear_iq_multi(self.iq_sweep_freqs,self.iq_sweep_data,self.tau)
        else:
            print("No iq sweep data found")

    def fit_power_sweep(self):
        if self.power_sweep_iq_data is not None:
            self.power_sweep_iq_fits = np.zeros((self.power_sweep_iq_data.shape[1],self.power_sweep_iq_data.shape[2],8))
            for k in tqdm(range(0,self.power_sweep_iq_data.shape[2])):
                self.power_sweep_iq_fits[:,k,:] = res_fit.fit_nonlinear_iq_multi(self.power_sweep_iq_freqs[:,:,k],
                                                                                 self.power_sweep_iq_data[:,:,k],self.tau)
        else:
            print("No power sweep data found")
                
    def tune_powers(self):
        if self.power_sweep_iq_fits is not None:
            self.picked_power_levels,self.normalizing_amplitudes = \
            tune_resonators.tune_resonance_power(self.power_sweep_iq_freqs,
                                                  self.power_sweep_iq_data,
                                                  self.attn_levels,
                                                  fitted_a_iq = self.power_sweep_iq_fits[:,:,4])
        else:
            print("Run fitting over power sweep first")
            
    def plot_iq_sweep_fits(self):
        if self.iq_sweep_data is not None:
            if self.iq_sweep_iq_fits is not None:
                ip = tune_resonators.interactive_plot(self.iq_sweep_freqs,self.iq_sweep_data,retune = False,
                                                  combined_data = self.iq_sweep_iq_fits,
                                                  combined_data_names = ["Resonator Frequencies (MHz)",
                                                                         "Qr","amp","phi","a","i0","q0","f0"])


            
