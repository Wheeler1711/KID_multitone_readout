import importlib
import matplotlib.pyplot as plt
import numpy as np
import qsghw.core.helpers as corehelpers
import qsghw.fpgaip.helpers as fpgaiphelpers
import qsghw.misc.windows as windows
import qsghw.core.packetparser as packetparser
#import packetparser as packetparser # This is for begugging only! 
import qsghw.core.interfaces as interfaces
import time
#from submm.instruments import hp83732a as hp

#from submm.instruments import BNC845 as bnc
#from submm.instruments import anritsu_mg3691a as an
#from submm.instruments import weinchel8310 as weinchel
from submm.lab_brick import core as corelabbrick
#from submm.instruments import cryocon22C as cc
import resonator_class as res_class
from submm.KIDs import find_resonances_interactive as find_res
from submm.KIDs import calibrate as cal
from submm.KIDs import analyze_multitone as am
from submm.KIDs.res import fitting as res_fit
from submm.KIDs.res import sweep_tools as tune_resonators
import os
import data_io as data_io
import glob
from tqdm import tqdm
from scipy import signal
from scipy import interpolate
yeses = ['yes','y']
import sys
import rfnco as rfnco
from matplotlib.animation import FuncAnimation
from scipy import signal # Added by AnV on 01/19/2024 for downsampling data before saving it 

class readout(object):

    def __init__(self,connect_rfsoc = True):
        '''
        readout class for interfacing with RFSOC firmware
        General procedue might be
        readout.vna_sweep(start,stop)
        readout.find_resonators()
        readout.write_resonator_tones()
        readout.iq_sweep()
        readout.power_sweep()
        readout.fit_power_sweep()
        readout.tune_powers()
        readout.write_resonator_tones()
        readout.iq_sweep()
        readout.input_attenuator.set_attenuation()
        readout.retune_resonators()
        readout.take_noise_set()
        To debug software without connecting to RFSOC set connect_rfsoc = False
        '''
        #self.ctrl = "tcp://192.168.0.128"
        self.mode = 'average'#'ttl'
        #self.mode = 'ttl'
        self.data_rate = 250000 # Changed by AnV 02/05/2024
        self.ctrl = "udp://10.0.15.11" 
        self.window_type = "roach2"
        #self.window_type = "hft70-1024"
        self.merge = 256
        self.decimate = 1
        self.amp = 0.4
        self.output = "FINEDDC"
        self.data = "Auto"
        self.open_data = True
        self.quiet = True
        self.insane = False
        self.regaccess = False
        self.show_access = False
        self.debug = False
        self.verbose = False
        self.group = 0
        self.fshift = -22
        self.pshift = 0
        self.qshift = -16
        self.pktsize = 8192
        self.tag = None
        self.frequencies = [-19.875*10**6,-50*10**6]
        self.amplitude = 0.4
        #self.lo = hp.synthesizer()
        #self.lo = an.synthesizer()
        self.data = "udp://10.0.15.11"
        self.iq_sweep_data = None
        self.iq_sweep_data_corr = None #IQ sweep data with cabel delay removed
        self.iq_sweep_data_old = None
        #self.input_attenuator = weinchel.attenuator('GPIB0::8::INSTR') #input to detectors, output of fpga
        #self.output_attenuator = None#weinchel.attenuator()
        self.input_attenuator = corelabbrick.Attenuator(0x41f,0x1208,11874)
        self.output_attenuator = corelabbrick.Attenuator(0x41f,0x1208,11875)
        #self.BB = cc.temp_control()
        self.res_class = res_class.resonators_class([])
        self.vna_data = None
        self.tau = 4377*10**-9
        self.stream_data = None
        self.ttl_data = None
        self.average_data= None
        #self.data_dir = "/data2/20231021_CCAT_array_2_optical_more_polcal_results/"
        #self.data_dir = "/data2/20231130_CCAT_array_1_trimmed/"
        #self.data_dir = "/data2/20240125_CCAT_array_1_optical/"
        #self.data_dir = "/data2/20240228_NbTiN_blackbody_and_Al_MKIDs/measurements_at_250mK/"
        self.data_dir = "/data/multitone_data/20250212_CCAT_led_map_280GHz_remake/"
        #self.data_dir = "/data2/20240521_Toltec_chip_polcal/"
        #self.data_dir = "/data2/20240815_EoRSpec_260GHz_array_1_LED_mapping/"
        #self.data_dir = "/data2/20241011_trimmed_CCAT_TiN_array_FTS/"
        #self.data_dir = "/data2/20240604_NbTiN_blackbody_at_10K/"
        self.data_dir = os.path.expanduser(self.data_dir)
        print(self.data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.vna_save_filename = "vna_sweep"
        self.iq_sweep_save_filename = "iq_sweep.p"
        self.stream_save_filename = "stream.p"
        self.res_class_save_filename = "resonators.csv"
        self.res_set = None
        self.power_sweep_iq_data = None
        self.power_sweep_iq_fits = None
        self.normalizing_amplitudes = None
        self.pbar_ascii = True

        self.folder_name = "" 
        
        if connect_rfsoc:
            (self.ctrlif, self.dataif, self.eids) = corehelpers.open(self.ctrl, self.data, opendata=self.open_data,
                                                                 verbose=self.debug, quiet=self.quiet)
            self.ctrlif.debug = [self.insane, self.show_access]

            self.lo = rfnco.rfnco("rfrtconfa",self.ctrlif,self.dataif,self.eids)
            self.lo_freq = 896.0*10**6/2#814.00*10**6
            #self.lo.set_frequency(self.lo_freq,fix_phase = False)
            # the other lo for doing sweeps
            self.fset = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":fullset:", 'fset0')
            self.tonegen = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":tonegen:", 'fset0')
            
            self.wfplayer = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":wfplayer:")
            #self.frontend = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":chpfbfft:")
            self.frontend = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, interfaces.channelizer)
            self.backend  = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, filt = 'backend')
            #self.circb = fpgaiphelpers.makeinstance(self.ctrlif, self.eids, ":circbuffer:", "")
            #self.wfplayer.debug = self.frontend.debug = self.frontend.pfbfft.debug = self.frontend.binsel[self.group].debug \
            #    = self.frontend.fineddc[self.group].debug = self.circb.debug = (self.regaccess, self.debug, self.verbose)
            self.wfplayer.debug = self.frontend.debug = self.frontend.pfbfft.debug = self.frontend.binsel[self.group].debug \
                = self.frontend.fineddc[self.group].debug = (self.regaccess, self.debug, self.verbose)
            #assert self.wfplayer and self.frontend and self.circb
            assert self.wfplayer and self.frontend #and self.circb

            
            self.samplerate = self.wfplayer.samplerate = self.frontend.get_samplerate()
            self.iqorder = self.wfplayer.iqorder = True if self.frontend.get_iqorder() else False
            print()
            print("samplerate = %8.3f MHz   iqorder = %d" % (self.samplerate/1e6, self.iqorder))
            print()

            (self.tonegen.samplerate, self.tonegen.iqorder) = (self.samplerate, self.iqorder)
            self.tonegen.set_i(amp = 0.5, mode = "ddstaylor-i", phase=0)
            self.tonegen.set_q(amp = 0.5, mode = "ddstaylor-q", phase=-90)
            print( self.tonegen.get_i(), self.tonegen.get_q() )
        
            # stop flow of data
            self.frontend.write_monstop(1)

            # generate and write window - accept exteneded window names
            # okay to overwirte window?
            self.window = windows.mkwinbyextname(self.window_type, self.frontend.get_windowlen(), self.frontend.get_fftlen())
            self.frontend.write_window(self.window)
            print("window =", self.window_type, len(self.window))

            #self.frontend.change_output(self.output,self.merge,self.decimate)
            self.frontend.write_fshift(self.fshift)
            self.frontend.write_pshift(self.pshift)
            self.frontend.write_qshift(self.qshift)
            print("bit shifts after filter pfbfft fineddc =", self.frontend.read_fshift(), self.frontend.read_pshift(),
                  self.frontend.read_qshift())
            print()

            # frontend output selection
            print("frontend output = %s   merge = %3d   decimate = %3d" % (self.output, self.merge, self.decimate))
            #self.frontend.change_output(self.output, self.merge, self.decimate)

            # apply tag
            if self.tag: self.frontend.write_tag(int(self.tag, 0))

            # not frontend, but anyways: program and reset circbuffer
            #self.circb.write_packetsize(self.pktsize)
            #self.circb.write_bufspace(0) # 0 => maximum supported
            #self.circb.reset()
            #print("circbuf packetsize = %5d   bufspace = %6d" % (self.circb.read_packetsize(), self.circb.read_bufspace()))

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


    def create_folder(self, folder_name = ""): # Added by AnV
        self.folder_name = folder_name; 
        if not os.path.exists(self.data_dir + self.folder_name):
            os.mkdir(self.data_dir + folder_name);
            print("Measured data will be saved to a new folder: " + folder_name);
        else:
            print("Measured data will be saved to an existing folder: " + folder_name);
            
            
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
        self.frequencies = frequencies
        if np.isscalar(amplitude): # supplying array or of amplitudes?
            amplitude = np.ones(len(frequencies))*amplitude
        if random_phase:
            np.random.seed()
            phase = np.random.uniform(0., 360, len(frequencies))
        else:
            phase = np.zeros(len(frequencies))

        self.phase = phase

        n_tones = len(frequencies)
        if self.mode == 'ttl': # packets are twice as big
            self.merge = 501//n_tones #how many samples in a packet
        else:
            self.merge = 1001//n_tones #how many samples in a packet   
        print("merge = "+str(self.merge))

        #if self.mode == 'ttl':
            #self.frontend.change_output('None')# this correct?
            #self.backend.change_output('INPUT',self.merge,self.decimate) # maybe
        #else:
        self.frontend.change_output(self.output, self.merge, self.decimate)
        self.backend.change_output('None')

        self.real_freq_list = []
        self.wfplayer.del_tone(groupid = self.group)
        for i in range(0,len(frequencies)):
            self.wfplayer.set_tone(self.group,i,frequencies[i],phase[i],phase[i],amplitude[i],amplitude[i])
            self.real_freq_list.append(self.wfplayer.get_tone(self.group,i)[0])

        self.wfplayer.write_tone_list()

        if fill_DAQ_range: # this is ineffecient currently writing tones to board twice
            maximum_of_wave = np.max((np.abs(np.real(self.wfplayer.wave)),np.abs(np.imag(self.wfplayer.wave))))
            scale_factor = (0.9*(2**16//2-1))/maximum_of_wave
            amplitude = amplitude*scale_factor
            self.amplitude = amplitude
            self.real_freq_list = []
            self.wfplayer.del_tone(groupid = self.group)
            for i in range(0,len(frequencies)):
                self.wfplayer.set_tone(self.group,i,frequencies[i],phase[i],phase[i],amplitude[i],amplitude[i])
                self.real_freq_list.append(self.wfplayer.get_tone(self.group,i)[0])
            self.wfplayer.write_tone_list()
            maximum_of_wave = np.max((np.abs(np.real(self.wfplayer.wave)),np.abs(np.imag(self.wfplayer.wave))))
            if maximum_of_wave>0.95*(2**16//2-1):
                print("!!!!Warning you may be clipping DAC waverform!!!!")

        # old way with list
        #self.real_freq_list = self.wfplayer.set_tone_list(self.group, frequencies, write=True, amp=amplitude[0])
        self.chanlist = self.frontend.set_chan_list(self.group, self.real_freq_list, write=True)
        self.binnolist = [ self.frontend.get_chan(self.group, i)[1] for i in self.chanlist ]

        print()
        print("     asked         actual       fftbin  channel")
        for i in range(len(frequencies)):
            print("  %8.3f MHz   %8.3f MHz   %6d   %6d" % (frequencies[i]/1e6, self.real_freq_list[i]/1e6,
                                                           self.binnolist[i], self.chanlist[i]))

        print()

    def write_scaled_frequencies(self,scale):
        amplitude = self.amplitude*scale
        self.real_freq_list = []
        self.wfplayer.del_tone(groupid = self.group)
        for i in range(0,len(self.frequencies)):
            self.wfplayer.set_tone(self.group,i,self.frequencies[i],self.phase[i],self.phase[i],amplitude[i],amplitude[i])
            self.real_freq_list.append(self.wfplayer.get_tone(self.group,i)[0])
        self.wfplayer.write_tone_list()
        maximum_of_wave = np.max((np.abs(np.real(self.wfplayer.wave)),np.abs(np.imag(self.wfplayer.wave))))
        print("max of wave = ",maximum_of_wave)

        self.chanlist = self.frontend.set_chan_list(self.group, self.real_freq_list, write=True)
        self.binnolist = [ self.frontend.get_chan(self.group, i)[1] for i in self.chanlist ]
            
    def iq_sweep(self,span = 100*10**3,npts = 101,average_time = 0.1,offset = 0.0,plot = True,leave = True,fit = False):
        if self.iq_sweep_data is not None:
            self.iq_sweep_data_old = self.iq_sweep_data
            self.iq_sweep_freqs_old = self.iq_sweep_freqs
        to_read = int(1024*8*average_time*self.data_rate/self.merge)//(1024*8)*1024*8
        #self.iq_sweep_freqs_lo = np.linspace(self.lo_freq-span/2+offset,self.lo_freq+span/2+offset,npts)
        self.iq_sweep_freqs_lo = np.linspace(-span/2+offset,span/2+offset,npts)
        self.iq_sweep_freqs = np.zeros((npts,len(self.real_freq_list)))
        #for m in range(0,len(self.real_freq_list)):
        #    self.iq_sweep_freqs[:,m] = self.iq_sweep_freqs_lo+self.real_freq_list[m]
        self.iq_sweep_data = np.zeros((len(self.iq_sweep_freqs),len(self.real_freq_list)),dtype ='complex')
        self.fset.write_mixmode(0xf25af2a5)

        actual_iq_sweep_freqs = np.asarray(())
        
        for n, nominalf in tqdm(enumerate(self.iq_sweep_freqs_lo),total = npts,leave = leave,ascii = self.pbar_ascii):
            self.tonegen.set_i(freq = nominalf); self.tonegen.set_q(freq = nominalf);
            time.sleep(0.01)
            curr_freq = self.tonegen.get_i()[0]#self.lo.get_frequency()                                                                                         
            actual_iq_sweep_freqs = np.append(actual_iq_sweep_freqs,curr_freq)
            #self.lo.set_frequency(freq,verbose = True)
            #time.sleep(0.1)
            #self.lo.get_frequency()
            alldata_array = self.get_data(to_read,verbose = False)
            for m in range(0,alldata_array.shape[1]):
                self.iq_sweep_data[n,m] = np.mean(alldata_array[:,m])   
                #self.iq_sweep_data[n,m] = np.mean(signal.decimate(alldata_array[:,m],10))


        for m in range(0,len(self.real_freq_list)):                                                                                                             
            self.iq_sweep_freqs[:,m] = self.lo_freq + actual_iq_sweep_freqs+self.real_freq_list[m]  

        if self.tau is not None:
            self.iq_sweep_data_corr = np.zeros(self.iq_sweep_data.shape,dtype = 'complex')
            for m in range(0,self.iq_sweep_data.shape[1]):
                self.iq_sweep_data_corr[:,m] = cal.remove_cable_delay(self.iq_sweep_freqs[:,m],self.iq_sweep_data[:,m],self.tau)
        #self.lo.set_frequency(self.lo_freq)
        #self.fset.write_mixmode(0x00010001)
        self.tonegen.set_i(freq = 0); self.tonegen.set_q(freq = 0);
        time.sleep(0.1)
        if fit:
            self.fit_iq_sweep()
        elif plot:
            self.plot_iq_sweep()

    def plot_iq_sweep(self,plot_two = False,retune = False):
        if self.iq_sweep_data is not None:
            if not plot_two:
                if self.iq_sweep_data_corr is not None:
                    ip = tune_resonators.InteractivePlot(self.iq_sweep_freqs,self.iq_sweep_data_corr,retune = retune,
                                                  combined_data = self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,:]/10**6,
                                                      combined_data_names = ["Resonator Frequencies (MHz)"],
                                                     flags = self.current_flags())
                else:
                    ip = tune_resonators.InteractivePlot(self.iq_sweep_freqs,self.iq_sweep_data,retune = retune,
                                                  combined_data = self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,:]/10**6,
                                                      combined_data_names = ["Resonator Frequencies (MHz)"],
                                                     flags = self.current_flags())
                self.update_flags(ip.flags)
            else:
                if self.iq_sweep_data_old is not None:
                    multi_sweep_freqs = np.dstack((np.expand_dims(self.iq_sweep_freqs,axis = 2),np.expand_dims(self.iq_sweep_freqs_old,axis = 2)))
                    multi_sweep_z = np.dstack((np.expand_dims(self.iq_sweep_data,axis = 2),np.expand_dims(self.iq_sweep_data_old,axis = 2)))
                    ip = tune_resonators.InteractivePlot(multi_sweep_freqs,multi_sweep_z,retune = retune,
                                                  combined_data = self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,:]/10**6,
                                                         combined_data_names = ["Resonator Frequencies (MHz)"],
                                                         flags = self.current_flags())
                    if ip.flags is not None:
                        self.update_flags(ip.flags)
                else:
                    print("only one iq sweep data set found cannot plot two")
        else:
            print("No IQ sweep data found: unable to plot")

    def retune_resonators(self,find_min = False,write_resonator_tones = True):
        if self.res_class.len_use_true() != self.iq_sweep_data.shape[1]:
            print("number of resonators in iq data does not match number of currently"+\
                  "active resonator cannot retune because iq information is ambiguous")
            return
        # could allow this if I do some matching to figure out what is what
        #    confirmation = input("number of resonators in iq data does not match"+\
        #                         "number of currently active resonators: Proceed? y/n")
        #    if confirmation in yeses: #should I allow this
        #        pass 
        #    else:
        #        return
        if self.iq_sweep_data_corr is not None:
            new_frequencies = tune_resonators.tune_kids(self.iq_sweep_freqs,self.iq_sweep_data_corr,find_min = find_min)
        else:
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


    def vna_sweep(self,start,stop,resolution = 1*10**3,average_time = 0.01,min_tone_spacing = 2.5*10**6, plot=True, set_freqs=True):
        '''
        Pretend we are vna for finding resonances
        currently limited to putting tone ~10MHz apart
        '''
        n_tones = int(stop-start)//(int(min_tone_spacing))+1
        if n_tones >1001:
            n_tones = 1001
        print("Using "+str(n_tones)+" for VNA sweep")
        self.frequencies = list(np.linspace(start-self.lo_freq,stop-self.lo_freq,n_tones))
        if 0 in self.frequencies: #don't put tone on LO
            n_tones = n_tones - 1
            self.frequencies = list(np.linspace(start-self.lo_freq,stop-self.lo_freq,n_tones))
        tone_spacing = self.frequencies[1]-self.frequencies[0]
        npts = int(tone_spacing//resolution)

        #iq_sweep_freqs = np.linspace(self.lo_freq-tone_spacing/2,self.lo_freq+tone_spacing/2,npts)
        iq_sweep_freqs = np.linspace(-tone_spacing/2,tone_spacing/2,npts) 
        if set_freqs:
            self.set_frequencies(self.frequencies,amplitude = 0.8/n_tones)

        self.vna_freqs = np.asarray(())
        #for m in range(0,len(self.real_freq_list)):
        #    self.vna_freqs = np.append(self.vna_freqs,self.real_freq_list[m]+iq_sweep_freqs)

        # 1 packet is 8k = 1024*8 each pa
        to_read = int(1024*8*average_time*self.data_rate/self.merge)//(1024*8)*1024*8 
        # at some point should turn this into an integration time
        print(to_read)
        iq_sweep_data = np.zeros((len(iq_sweep_freqs),len(self.real_freq_list)),dtype ='complex')

        pbar = tqdm(enumerate(iq_sweep_freqs),total = len(iq_sweep_freqs),ascii=self.pbar_ascii)

        self.fset.write_mixmode(0xf25af2a5)

        actual_iq_sweep_freqs = np.asarray(())
        for n, nominalf in pbar:
            #self.lo.set_frequency(freq)
            #print(nominalf)
            self.tonegen.set_i(freq = nominalf); self.tonegen.set_q(freq = nominalf);
            time.sleep(0.001)
            curr_freq = self.tonegen.get_i()[0]#self.lo.get_frequency()
            actual_iq_sweep_freqs = np.append(actual_iq_sweep_freqs,curr_freq)
            pbar.set_description(f'{"%3.3f" % (curr_freq/10**6)} MHz')
            #print(str(curr_freq/10**6)+" MHz")
            alldata_array = self.get_data(to_read,verbose = False)
            for m in range(0,alldata_array.shape[1]):
                iq_sweep_data[n,m] = np.mean(alldata_array[:,m])

        #self.fset.write_mixmode(0x00010001)
                
        for m in range(0,len(self.real_freq_list)):                                                                                                            
            self.vna_freqs = np.append(self.vna_freqs,self.lo_freq+self.real_freq_list[m]+actual_iq_sweep_freqs)
        

        self.vna_data = np.asarray(())
        for m in range(0,len(self.real_freq_list)):
            self.vna_data = np.append(self.vna_data,iq_sweep_data[:,m])

        if plot:
            self.plot_vna_sweep()


    def get_stream_data(self,stream_time= 60):
        #self.lo.set_frequency(self.lo_freq) #make sure LO is back at nominal frequency
        time.sleep(0.1)
        stream_len = int(1024*8*stream_time*self.data_rate/self.merge)//(1024*8)*1024*8
        self.stream_frequencies = [freq + self.lo_freq for freq in self.real_freq_list]
        if self.mode == 'ttl':
            self.stream_data, self.ttl_data  = self.get_data(stream_len,verbose = False,ttl = True)
        elif self.mode == 'average':
            self.stream_data, self.average_data = self.get_data(stream_len,verbose = False,average = True)
        else:
            self.stream_data = self.get_data(stream_len,verbose = False)


    def plot_iq_and_stream(self,decimate = 1000):#eventualy decimate -> frequency
        # decimate only handels 1 sig fig i.e 2000 no 2500
        if self.stream_data is not None:
            if self.iq_sweep_data is not None:
                self.decimated_stream_data = self.stream_data #need deep copy?
                #factors_of_10 = int(np.floor(np.log10(decimate)))
                #for k in range(0,factors_of_10):
                #    self.decimated_stream_data = signal.decimate(self.decimated_stream_data,10,axis = 0)
                #self.decimated_stream_data = signal.decimate(self.decimated_stream_data,
                #                                             decimate//(10**factors_of_10),axis =0)
                #decimate seems to bias the mean
                self.decimated_stream_data = signal.resample_poly(self.decimated_stream_data,1,decimate,padtype = 'mean')

                ip = tune_resonators.InteractivePlot(self.iq_sweep_freqs,self.iq_sweep_data,
                                                      stream_data = self.decimated_stream_data,
                                                      retune = False,flags = self.current_flags())
                self.update_flags(ip.flags)
            else:
                print("No IQ sweep data found: unable to plot")
        else:
            print("No stream data found: unable to plot")


    def plot_vna_sweep(self):
        plt.figure(1)
        plt.plot(self.vna_freqs/10**6,20*np.log10(np.abs(self.vna_data)))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Power (dB)")
        plt.title("VNA Sweep")
        plt.show()

    def save_vna_data(self,save_as_pickle = False):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.vna_save_filename
        data_io.write_vna_data(save_filename,self.vna_freqs,self.vna_data,save_as_pickle = save_as_pickle)

    def load_vna_data(self,filename = None):
        if not filename: # look for most recent file
            list_of_files = glob.glob(self.data_dir+'/*'+self.vna_save_filename+'*')
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
        save_filename = self.data_dir + "/" + self.folder_name + "/" +timestr+"_"+self.iq_sweep_save_filename.split(".")[0]+\
            filename_suffix+"."+self.iq_sweep_save_filename.split(".")[1] # Anna
        data_io.write_iq_sweep_data(save_filename,self.iq_sweep_freqs,self.iq_sweep_data)
        return save_filename

    def load_iq_data(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.iq_sweep_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.iq_sweep_freqs,self.iq_sweep_data = data_io.read_iq_sweep_data(latest_file)
        else:
            try:
                if self.iq_sweep_data is not None:
                    self.iq_sweep_data_old = self.iq_sweep_data
                    self.iq_sweep_freqs_old = self.iq_sweep_freqs
                self.iq_sweep_freqs,self.iq_sweep_data = data_io.read_iq_sweep_data(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)

    def save_stream_data(self,filename_suffix = "", folder_name = "Test", downsampling = 250):
        # Added a new downsampling parameter downsampling  = 250 (default -> 1 kHz sampling rate), by AnV 01/19/2024 
        timestr = time.strftime("%Y%m%d_%H%M%S");
        save_filename = self.data_dir + "/" + self.folder_name + "/" + timestr+"_"+self.stream_save_filename.split(".")[0]+\
            filename_suffix+"."+self.stream_save_filename.split(".")[1]
        stream_data_downsampled = signal.resample_poly(self.stream_data, 1, downsampling, padtype = 'mean'); 
        if self.mode == 'ttl':
            ttl_data_downsampled = np.round(signal.resample_poly(self.ttl_data.astype('float'), 1, downsampling, padtype = 'constant'));
            data_io.write_stream_data(save_filename,self.stream_frequencies,stream_data_downsampled,ttl_data = ttl_data_downsampled, downsampling = downsampling);
            # Downsampling also added to data_io.write ... function by AnV on 01/19/2024
        else:
            data_io.write_stream_data(save_filename,self.stream_frequencies,stream_data_downsampled, downsampling = downsampling);
        return save_filename

    def load_stream_data(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.stream_save_filename)
            latest_file = max(list_of_files, key=os.path.getctime)
            self.stream_freqs,self.stream_data = data_io.read_stream_data(latest_file)
        else:
            try:
                if self.mode == 'ttl':
                    self.stream_frequencies,self.stream_data,self.ttl_data = data_io.read_stream_data(filename)
                else:
                    self.stream_frequencies,self.stream_data = data_io.read_stream_data(filename)
            except:
                print("could not read file: "+filename)
                if not os.path.exists(filename):
                    print(filename+" does not seem to exist")
                else:
                    print("error with file "+filename)
                    
    def find_resonators(self):
        if list(self.vna_data): #pythonic booleaness of list/None does not work on np.arrays() 
            if len(self.res_class.resonators)==0: #first time finding resonators
                ip = find_res.find_vna_sweep(self.vna_freqs,self.vna_data)
                for m in range(len(ip.kid_idx)):
                    self.res_class.resonators.append(res_class.resonator(ip.chan_freqs[ip.kid_idx[m]],
                                                                         use = True))
                    self.res_class.resonators[m].flags = ip.flags[m]
            else:
                print("Code to handle retunning from vna")
                self.res_class = resonator_class.resonators_class([]);
                ip = find_res.find_vna_sweep(self.vna_freqs,self.vna_data)
                for m in range(len(ip.kid_idx)):
                    self.res_class.resonators.append(res_class.resonator(ip.chan_freqs[ip.kid_idx[m]],
                                                                         use = True))
                    self.res_class.resonators[m].flags = ip.flags[m]
                        
        else:
            print("No VNA data found")

    def save_res_class(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir+"/"+timestr+"_"+self.res_class_save_filename
        data_io.write_resonators_class(save_filename,self.res_class)

    def load_res_class(self,filename = None):
        if not filename:
            list_of_files = glob.glob(self.data_dir+'/*'+self.res_class_save_filename)
            print(list_of_files) # AnV
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
        
            

    def get_data(self,to_read,verbose= True,ttl = False,average = False): 
        self.dataif.reset()
        (nbytes,raw_data) = self.dataif.readall(to_read)
        #print(nbytes)
        if verbose:
            print("raw data",sys.getsizeof(raw_data))
        #(headeroffsets, payloadoffsets, payloadlengths, seqnos) = \
        #    packetparser.findlongestcontinuous(raw_data, incr=512)
        
        (headeroffsets, payloadoffsets, payloadlengths, seqnos) = \
            packetparser.findlongestcontinuous(raw_data, incr=512, srcid = 142999584) # Added by AnV 02/01/2024

        
        if verbose:
            print(sys.getsizeof(headeroffsets),sys.getsizeof(payloadoffsets),sys.getsizeof(payloadlengths),sys.getsizeof(seqnos))
        (consumed, alldata) = packetparser.parsemany(raw_data, headeroffsets[0], payloadoffsets, verbose = verbose)
        if verbose:
            print(sys.getsizeof(consumed),sys.getsizeof(alldata))

        alldata_array = np.zeros(alldata.data.shape,dtype = 'complex')
        alldata_array = alldata.data['i']+1j*alldata.data['q']
        if average:
            (headeroffsets, payloadoffsets, payloadlengths, seqnos) = \
                                packetparser.findlongestcontinuous(raw_data, incr=512, srcid = 149422176) #
            (avearge_consumed, average_alldata) = packetparser.parsemany(raw_data, headeroffsets[0], payloadoffsets, verbose = verbose)
            average_alldata_array = np.zeros(average_alldata.data.shape,dtype = 'complex')
            average_alldata_array = average_alldata.data['i']+1j*average_alldata.data['q']
            return alldata_array,average_alldata_array 
        if ttl:
            
            (headeroffsets, payloadoffsets, payloadlengths, seqnos) = \
                packetparser.findlongestcontinuous(raw_data, incr=512, srcid = 101187616) # Added by AnV 02/01/2024
            
            if verbose:
                print(sys.getsizeof(headeroffsets),sys.getsizeof(payloadoffsets),sys.getsizeof(payloadlengths),sys.getsizeof(seqnos))
            (ttlconsumed, ttldata) = packetparser.parsemany(raw_data, headeroffsets[0], payloadoffsets, verbose = verbose)
            
            ttl_value_list = [];
            ttl_time_list = []; 
            for i in range(0, ttldata.data.shape[0]):
                #if ttldata.data[i]['active'] == 2147483648 or ttldata.data[i]['active'] == 3221225472: # Do not use
                ttl_value_list.append(float(bin(ttldata.data[i]['value'][0])[-1]))
                ttl_time_list.append(float(ttldata.data[i]['t'][0])*10000/2560*pow(10,-9)-alldata.timestamp) 
                       
            if verbose:
                print(sys.getsizeof(consumed),sys.getsizeof(alldata)) 


            ttl_value = np.asarray(ttl_value_list)
            readout.ttl_value = ttl_value
            ttl_time = np.asarray(ttl_time_list)
            ttl_time = ttl_time
            readout.ttl_time = ttl_time

            ttl_interp = interpolate.interp1d(ttl_time, ttl_value, kind='previous',bounds_error = False,fill_value = "extrapolate");
            #ttl_array = ttl_interp(np.linspace(ttl_time[0], ttl_time[-1], len(alldata_array.data)));
            ttl_array = ttl_interp(np.linspace(0,alldata.data.shape[0]/self.data_rate,alldata.data.shape[0]))
            
            # Testing 
            #self.alldata_array = alldata_array
            #self.ttl_array = ttl_array
            
            return alldata_array, ttl_array
        if verbose:
            print(sys.getsizeof(alldata_array))
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
            #print(0.8/len(frequencies))
            #print(np.isscalar(0.8/len(frequencies)))
            self.set_frequencies(frequencies,0.8/len(frequencies))

    def mask_by_separation(self,required_separation,mask_flags = False):
        '''
        turn off tones that are too close together
        '''
        last_active_frequency = None
        for resonator in sorted(self.res_class.resonators):
            if not last_active_frequency:#handle first res
                #print("First resonator")
                if not mask_flags:
                    last_active_frequency = sorted(self.res_class.resonators)[0].frequency
                    resonator.use = True
                elif len(resonator.flags)>0:
                    resonator.use = False
                else:
                    last_active_frequency = sorted(self.res_class.resonators)[0].frequency
                    resonator.use = True
            elif resonator.frequency<last_active_frequency+required_separation:
                print("Turning resonator off")
                resonator.use = False
            else:
                if not mask_flags:
                    print("Leaving resonator on")
                    resonator.use = True
                    last_active_frequency = resonator.frequency
                elif len(resonator.flags)>0:
                    resonator.use = False
                else:
                    resonator.use = True
                    last_active_frequency = resonator.frequency

    def mask_collisions(self,required_separation,mask_flags = False):
        '''                                                                                                                                                     
        turn off tones that are too close together                                                                                                              
        '''
        frequencies = np.asarray(self.res_class.all_frequencies())
        distance_left = np.abs(frequencies - np.roll(frequencies,-1))
        distance_right = np.abs(frequencies - np.roll(frequencies,1))
        for i, resonator in enumerate(self.res_class.resonators):
            if distance_left[i] < required_separation:
                resonator.use = False
                print(resonator.frequency,"Turning resonator off")
            elif distance_right[i] < required_separation:
                resonator.use = False
                print(resonator.frequency,"Turning resonator off")
            elif mask_flags and len(resonator.flags)>0:
                resonator.use = False
                print(resonator.frequency,"Turning resonator off")
            else:
                resonator.use = True
                print(resonator.frequency,"Turning resonator on")
   
                    
    def take_noise_set(self,fine_span = 100e3,stream_time= 60,average_time = 0.1,plot = True,retune = True,filename_suffix = "",downsampling=250):
        filenames = []
        if retune:
            print("Taking tuning iq sweep")
            self.iq_sweep(fine_span,average_time = average_time,plot = False)
            self.retune_resonators()
        #print("taking gain sweep")
        #self.iq_sweep(gain_span,plot = False)
        #filenames.append(self.save_iq_data())
        print("taking iq sweep")
        self.iq_sweep(fine_span,average_time = average_time,plot = False)
        filenames.append(self.save_iq_data(filename_suffix = filename_suffix))
        print("taking streaming data")
        self.get_stream_data(stream_time)
        filenames.append(self.save_stream_data(filename_suffix = filename_suffix,folder_name = "Test", downsampling = downsampling)) # Changed by AnV 01/26/2024
        self.save_noise_set_filenames(filenames,filename_suffix = filename_suffix)
        if plot:
            self.plot_iq_and_stream()
        

    def save_noise_set_filenames(self,filenames,filename_suffix): # AnV
        timestr = time.strftime("%Y%m%d_%H%M%S")
        save_filename = self.data_dir + "/" + self.folder_name +"/"+timestr+"_noise_set_filenames"+filename_suffix+".txt"
        data_io.write_filename_set(save_filename,filenames)
        

    def fit_cable_delay(self,plot = True):
        if list(self.vna_data):
            phase = np.arctan2(np.real(self.vna_data),np.imag(self.vna_data))
            self.tau,phase_gradient = cal.fit_cable_delay_from_slope(self.vna_freqs,phase,plot = plot)
            print("tau = "+str(self.tau))
        else:
            print("No VNA data found")


    def power_sweep(self,max_attenuation,min_attenuation,attenuation_step,span=100*10**3,npts = 101,
                    average_time = 0.7,output_attenuation = None,plot = False,pause_between_powers =1 ):
        '''
        do iq sweeps at different power levels to find bifurcation power levels
        output_attenutaion should be set to a single value that is good is at 
        halfway between max and min attenuation
        '''
        n_attn_levels = int((max_attenuation-min_attenuation)/attenuation_step)+1
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
        pbar = tqdm(range(0,n_attn_levels),ascii = self.pbar_ascii)
        initial_input_attenuation = self.input_attenuator.get_attenuation()
        if self.output_attenuator is not None:
            initial_output_attenuation = self.output_attenuator.get_attenuation()
        for k in pbar:
            self.input_attenuator.set_attenuation(attn_levels[k])
            input_attenuation = self.input_attenuator.get_attenuation()
            if self.output_attenuator is not None:
                self.output_attenuator.set_attenuation(attn_levels_output[k])
                output_attenuation = self.output_attenuator.get_attenuation()
                pbar.set_description("Input Attenuation = "+str(input_attenuation)+" Output_attenaution = " +\
                      str(output_attenuation))
            else:
                pbar.set_description("Input Attenuation = "+str(input_attenuation))
            time.sleep(pause_between_powers)
            self.iq_sweep(span,npts,average_time,plot = False,leave = False)
            self.power_sweep_iq_data[:,:,k] = self.iq_sweep_data
            self.power_sweep_iq_freqs[:,:,k] = self.iq_sweep_freqs
            filenames.append(self.save_iq_data())
        self.attn_levels = attn_levels
        self.save_power_sweep_filenames(filenames,attn_levels)
        self.input_attenuator.set_attenuation(initial_input_attenuation)
        if self.output_attenuator is not None:
            self.output_attenuator.set_attenuation(initial_output_attenuation)


    def dac_power_sweep(self,max_attenuation,min_attenuation,attenuation_step,span=100*10**3,npts = 101,
                         average_time = 0.1,output_attenuation = None,plot = False,pause_between_powers =1 ):
        '''                           
        same as power sweep but we are not going to use the output attenuator instead we are just going 
        to change the power by scaling the daq wave form
        do iq sweeps at different power levels to find bifurcation power levels                             
        halfway between max and min attenuation                                           
        '''
        n_attn_levels = int((max_attenuation-min_attenuation)/attenuation_step)+1
        attn_levels = np.linspace(max_attenuation,min_attenuation,n_attn_levels)
        voltage_scale = 10**(-attn_levels/20)
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
        pbar = tqdm(range(0,n_attn_levels),ascii = self.pbar_ascii)
        initial_input_attenuation = 0#self.input_attenuator.get_attenuation()
        if self.output_attenuator is not None:
            initial_output_attenuation = self.output_attenuator.get_attenuation()
        self.write_resonator_tones()
        for k in pbar:
            self.write_scaled_frequencies(scale = voltage_scale[k])
            #self.input_attenuator.set_attenuation(attn_levels[k])
            input_attenuation = self.input_attenuator.get_attenuation()
            if self.output_attenuator is not None:
                self.output_attenuator.set_attenuation(attn_levels_output[k])
                output_attenuation = self.output_attenuator.get_attenuation()
                pbar.set_description("Input Attenuation = "+str(attn_levels[k])+" Output_attenaution = " +\
                      str(output_attenuation))
            else:
                pbar.set_description("Input Attenuation = "+str(attn_levels[k]))
            time.sleep(pause_between_powers)
            self.iq_sweep(span,npts,average_time,plot = False,leave = False)
            self.power_sweep_iq_data[:,:,k] = self.iq_sweep_data
            self.power_sweep_iq_freqs[:,:,k] = self.iq_sweep_freqs
            filenames.append(self.save_iq_data())
        self.attn_levels = attn_levels
        self.save_power_sweep_filenames(filenames,attn_levels)
        self.input_attenuator.set_attenuation(initial_input_attenuation)
        if self.output_attenuator is not None:
            self.output_attenuator.set_attenuation(initial_output_attenuation)            
        
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
            
    def fit_iq_sweep(self,plot = True):
        if self.iq_sweep_data is not None:
            if self.res_class is not None:
                center_freqs = []
                for resonator in self.res_class.resonators:
                    center_freqs.append(resonator.frequency)
            else:
                center_freqs = None

            self.res_set = res_fit.fit_nonlinear_iq_multi(self.iq_sweep_freqs,
                                                                   self.iq_sweep_data,
                                                                   tau = self.tau,
                                                                   center_freqs = np.asarray(center_freqs),
                                                                   fit_overlap = 1/3.)
            if plot:
                self.plot_iq_sweep_fits()
        else:
            print("No iq sweep data found")

    def fit_power_sweep(self):
        if self.power_sweep_iq_data is not None:
            self.power_sweep_iq_fits = np.zeros((self.power_sweep_iq_data.shape[1],self.power_sweep_iq_data.shape[2],9))
            self.power_sweep_fit_iq_data = np.zeros(self.power_sweep_iq_data.shape,dtype=complex)
            if self.res_class is not None:
                center_freqs = []
                for resonator in self.res_class.resonators:
                    center_freqs.append(resonator.frequency)
                center_freqs = np.asarray(center_freqs)
            else:
                center_freqs = None
            for k in tqdm(range(0,self.power_sweep_iq_data.shape[2]),ascii = self.pbar_ascii):
                res_set = res_fit.fit_nonlinear_iq_multi(self.power_sweep_iq_freqs[:,:,k],
                                                           self.power_sweep_iq_data[:,:,k],
                                                           tau = self.tau,center_freqs = center_freqs,
                                                           verbose = False)
                for m, result in enumerate(res_set):
                    self.power_sweep_iq_fits[m,k,:] = result[0:9]
                    self.power_sweep_fit_iq_data[:,m,k] = res_set._fit_results[result].z_fit()
        else:
            print("No power sweep data found")
                
    def tune_powers(self):
        if self.power_sweep_iq_fits is not None:
            self.picked_power_levels,self.normalizing_amplitudes = \
            tune_resonators.tune_resonance_power(self.power_sweep_iq_freqs,
                                                 self.power_sweep_iq_data,
                                                 self.attn_levels,
                                                 fitted_a_iq = self.power_sweep_iq_fits[:,:,4],
                                                 z_fit_mag = self.power_sweep_fit_iq_data**2,
                                                 z_fit_iq = self.power_sweep_fit_iq_data)
        else:
            print("Run fitting over power sweep first")
            
    def plot_iq_sweep_fits(self):
        if self.iq_sweep_data is not None:
            if self.res_set is not None:
                ip  = self.res_set.plot(flags = self.current_flags())
                self.update_flags(ip.flags)
            else:
                print("No fit data found")
        else:
            print("No iq data found")

    def current_flags(self):
        if len(self.res_class.resonators) > 0:
            current_flags = []
            for resonator in self.res_class.resonators:
                if resonator.use:
                    current_flags.append(resonator.flags)
            return current_flags
        else:
            print("no resonator class found")
            return None

    def update_flags(self,flags):
        if self.res_class is not None and len(flags) > 0:
            for i, resonator in enumerate(self.res_class.resonators):
                if resonator.use:
                    resonator.flags = flags[i]
        else:
            print("no resonator class found or no flags found")

            
    def quick_power_tune(self,range_limit =10):
        if self.normalizing_amplitudes is None:
            self.normalizing_amplitudes = np.ones(self.iq_sweep_data.shape[1])
        pow_factor = []
        for i, fit in enumerate(self.res_set):
            if fit.a >0.1:
                pow_factor.append(0.5/fit.a)
            else:
                pow_factor.append(5) # if two low turn up power by 3dB
        pow_factor = np.asarray(pow_factor)
        median_pow_factor = np.median(pow_factor)
        index_high = np.where(pow_factor>median_pow_factor*np.sqrt(range_limit))
        pow_factor[index_high] = median_pow_factor*range_limit
        self.normalizing_amplitudes = self.normalizing_amplitudes*np.sqrt(pow_factor)

    def monitor(self,span = 100e3,average_time = 0.1):
        '''
        A function for monitoring the signal in real time
        probably should make put this in a seperate file as it own class
        '''
        self.iq_sweep(span = span,plot = False)
        self.cal_dict = am.calibrate_multi(self.iq_sweep_freqs,self.iq_sweep_data,tau = self.tau,plot = False)

        self.monitor_fig = plt.figure(figsize = (12,8))
        self.monitor_ax_iq = self.monitor_fig.add_subplot(221)
        self.monitor_ax_iq.set_aspect('equal')
        self.monitor_ax_df_over_f = self.monitor_fig.add_subplot(222)
        self.monitor_ax_all_df_over_f = self.monitor_fig.add_subplot(212)
        
        self.curr_m = 0
        def on_key_press(event):
            if event.key == 'left':
                if self.curr_m != 0:
                    self.curr_m = self.curr_m -1
                else:
                    self.curr_m = int(self.iq_sweep_freqs.shape[1]-1)
            if event.key == 'right':
                if self.curr_m != int(self.iq_sweep_freqs.shape[1]-1):
                    self.curr_m = self.curr_m +1
                else:
                    self.curr_m = 0
        self.monitor_fig.canvas.mpl_connect('key_press_event', on_key_press)
        x_data = []
        y_data = []
        self.monitor_df_over_f, = self.monitor_ax_df_over_f.plot(x_data, y_data, '-')
        self.monitor_iq_sweep, = self.monitor_ax_iq.plot(np.real(self.cal_dict['fine_corr'][:,0]),
                                                         np.imag(self.cal_dict['fine_corr'][:,0]),'-o')
        #self.interval = interval
        self.monitor_ax_df_over_f.set_title('df/f Resonator 0')
        self.monitor_ax_df_over_f.set_xlabel('time (s)')
        self.monitor_ax_df_over_f.set_ylabel('df/f')

        self.monitor_ax_all_df_over_f.set_xlabel('Resonator index')
        self.monitor_ax_all_df_over_f.set_ylabel('df/f')
        

        stream_len = int(1024*8*average_time*self.data_rate/self.merge)//(1024*8)*1024*8
        alldata_array = self.get_data(stream_len,verbose = False)
        self.then = time.time()
        self.times = np.zeros(1)
        self.iq_data_w_time = np.zeros(alldata_array.shape[1])
        self.iq_df_over_f_w_time = np.zeros(alldata_array.shape[1])
        self.all_z_corr = np.zeros(alldata_array.shape[1],dtype = 'complex')
        for m in range(0,alldata_array.shape[1]):
            z = np.mean(alldata_array[:,m])
            z_corr = cal.remove_cable_delay(self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,m],z,self.tau)
            z_corr = z_corr - self.cal_dict['circle_fit'][m,0] -1j*self.cal_dict['circle_fit'][m,1]
            z_corr = z_corr*np.exp(-1j*self.cal_dict['circle_fit'][m,-1])
            self.all_z_corr[m] = z_corr
            self.iq_data_w_time[m] = self.cal_dict['interp_functions'][m](np.arctan2(np.real(z_corr),np.imag(z_corr)))
        #y_data.append(self.iq_data_w_time[curr_m])
        self.monitor_iq_point, = self.monitor_ax_iq.plot(np.real(self.all_z_corr[0]),np.imag(self.all_z_corr[0]),"*")
        self.monitor_all_df_over_f, = self.monitor_ax_all_df_over_f.plot(np.arange(0,len(self.iq_df_over_f_w_time)),
                                                                         self.iq_df_over_f_w_time,"o")
        self.monitor_all_df_over_f_highlight, = self.monitor_ax_all_df_over_f.plot(0,self.iq_df_over_f_w_time[0],'o')
        
        def update(frame):
            self.times = np.append(self.times,time.time()-self.then)
            alldata_array = self.get_data(stream_len,verbose = False)
            iq_data_w_time_temp = np.zeros(alldata_array.shape[1])
            for m in range(0,alldata_array.shape[1]):
                z =	np.mean(alldata_array[:,m])
                z_corr = cal.remove_cable_delay(self.iq_sweep_freqs[self.iq_sweep_freqs.shape[0]//2,m],z,self.tau)
                z_corr = z_corr - self.cal_dict['circle_fit'][m,0] -1j*self.cal_dict['circle_fit'][m,1]
                z_corr = z_corr*np.exp(-1j*self.cal_dict['circle_fit'][m,-1])
                self.all_z_corr[m] = z_corr
                iq_data_w_time_temp[m] = self.cal_dict['interp_functions'][m](np.arctan2(np.real(z_corr),np.imag(z_corr)))

            self.iq_data_w_time = np.vstack((self.iq_data_w_time,iq_data_w_time_temp))
            self.iq_df_over_f_w_time = (self.iq_data_w_time-self.iq_data_w_time[0,:])/self.iq_data_w_time[0,:]

                                 
            self.monitor_df_over_f.set_data(self.times,self.iq_df_over_f_w_time[:,self.curr_m] )

            # all df over f plot
            self.monitor_all_df_over_f.set_data(np.arange(0,self.iq_df_over_f_w_time.shape[1]),
                                                self.iq_df_over_f_w_time[-1,:])
            self.monitor_all_df_over_f_highlight.set_data(self.curr_m,self.iq_df_over_f_w_time[-1,self.curr_m])

            # iq circle plot
            self.monitor_iq_sweep.set_data(np.real(self.cal_dict['fine_corr'][:,self.curr_m]),
                                           np.imag(self.cal_dict['fine_corr'][:,self.curr_m]))
            self.monitor_iq_point.set_data(np.real(self.all_z_corr[self.curr_m]),np.imag(self.all_z_corr[self.curr_m]))
            
            self.monitor_ax_df_over_f.relim()
            self.monitor_ax_df_over_f.autoscale_view()

            self.monitor_ax_all_df_over_f.set_ylim(np.min(self.iq_df_over_f_w_time),np.max(self.iq_df_over_f_w_time))
            #self.monitor_ax_all_df_over_f.autoscale_view()

            self.monitor_ax_iq.relim()
            self.monitor_ax_iq.autoscale_view()
            #plt.plot(x_data, y_data, '-', color='blue')
            self.monitor_ax_df_over_f.set_title('df/f Resonator '+str(self.curr_m))
            return self.monitor_df_over_f,
        self.monitor_iq_sweep,
        self.monitor_iq_point,
        self.monitor_all_df_over_f,
        self.monitor_all_df_over_f_highlight

 
        animation = FuncAnimation(self.monitor_fig, update)
        plt.show()
