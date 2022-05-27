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
import tune_resonators as tune_resonators
import os
import data_io as data_io
import glob
from tqdm import tqdm
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
        self.res_class = res_class.resonators_class([])
        self.vna_data = None
        self.data_dir = "~/multitone_data"
        self.data_dir = os.path.expanduser(self.data_dir)
        print(self.data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.vna_save_filename = "vna_sweep.csv"

        
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
        self.merge = 512//n_tones
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


    def iq_sweep(self,span = 100*10**3,npts = 101,average = 1024*10,offset = 0.0):
        to_read = int(1024*average) # at some point should turn this into an integration time
        self.iq_sweep_freqs_lo = np.linspace(self.lo_freq-span/2+offset,self.lo_freq+span/2+offset,npts)
        self.iq_sweep_freqs = np.zeros((npts,len(self.real_freq_list)))
        for m in range(0,len(self.real_freq_list)):
            self.iq_sweep_freqs[:,m] = self.iq_sweep_freqs_lo+self.real_freq_list[m]
        self.iq_sweep_data = np.zeros((len(self.iq_sweep_freqs),len(self.real_freq_list)),dtype ='complex')
        for n, freq in tqdm(enumerate(self.iq_sweep_freqs_lo),total = npts):
            self.lo.set_frequency(freq)
            time.sleep(0.01)
            self.lo.get_frequency()
            alldata_array = self.get_data(to_read,verbose = False)
            for m in range(0,alldata_array.shape[1]):
                self.iq_sweep_data[n,m] = np.mean(alldata_array[:,m])

    def plot_iq_sweep(self):
        if self.iq_sweep_data is not None:
            ip = tune_resonators.interactive_plot(self.iq_sweep_freqs,self.iq_sweep_data,retune = False)        
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


    def write_resonator_tones(self):
        frequencies = []
        for m in range(len(self.res_class.resonators)):
            if self.res_class.resonators[m].use == True:
                frequencies.append(self.res_class.resonators[m].frequency-self.lo_freq)

        self.set_frequencies(frequencies,0.8)

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
