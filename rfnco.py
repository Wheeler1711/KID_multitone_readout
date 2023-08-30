import qsghw.fpgaip.helpers as fpgaiphelpers
import time

class rfnco(object):

    def __init__(self,path,ctrlif,dataif,eids):
        self.rfrtconf = fpgaiphelpers.makeinstance(ctrlif,eids,":rfrtconf",path)
        
    def get_frequency(self):
        #print("the current frequency is: ")
        return(600e6)
    def set_frequency(self,freq,verbose = False,fix_phase = False):

        if verbose:
            self.xprint(self.rfrtconf, "before:")

        if fix_phase:
            for i in range(128):
                # assume there's one DAC NCO @ 0, and one ADC NCO @ 1                                                             
                dacactual = self.rfrtconf.lo[0].set_freq(freq)
                adcactual = self.rfrtconf.lo[1].set_freq(-freq)
                time.sleep(0.05)
                dac1 = self.rfrtconf.read_dacbusyfall() - self.rfrtconf.read_dacreqrise()
                adc1 = self.rfrtconf.read_adcbusyfall() - self.rfrtconf.read_adcreqrise()

                # reset synchronously the DAC and the ADC NO phase                                                                
                self.rfrtconf.rstphase()
                time.sleep(0.05)
                dac2 = self.rfrtconf.read_dacbusyfall() - self.rfrtconf.read_dacreqrise()
                adc2 = self.rfrtconf.read_adcbusyfall() - self.rfrtconf.read_adcreqrise()

                # check how long it took for the NCO updates to complete, exit loop if it's the golden values                     
                #    print("--", dac1, adc1, dac2, adc2)                                                                              
                #    if dac1 == 42 and adc1 == 68 and dac2 == 26 and adc2 == 40: break # 37373737                                     
                if dac1 == 40 and adc1 == 68 and dac2 == 26 and adc2 == 40: break # 1f1f1f1f                                      
            assert i < 127
        else:
            dacactual = self.rfrtconf.lo[0].set_freq(freq)
            adcactual = self.rfrtconf.lo[1].set_freq(-freq)
            time.sleep(0.1)
            self.rfrtconf.rstphase()
            
        # assume there's one DAC NCO @ 0, and one ADC NCO @ 1                                                         
        #dacactual = self.rfrtconf.lo[0].set_freq(freq)
        #adcactual = self.rfrtconf.lo[1].set_freq(-freq)
        #self.rfrtconf.rstphase()

        if verbose:
            self.xprint(self.rfrtconf, "after:")
        

    def xprint(self,rfrtconf, note = None):
        print(note)
        for osc in rfrtconf.lo:
            n = osc.get_freq_raw()
            pn = n              if n != None else -1
            pf = osc.get_freq() if n != None else -1
            ref = osc.get_ref_freq()
            (num, denom) = osc.get_freq_res_asfrac()
            print("%25s.%-8s %2d   %12s %12.6f     %25s   %12x" % (rfrtconf.entityid.path, osc.name, osc.id,
                                                                   format(ref, ",.0f"), (num/denom), format(pf, ",.9f"), pn))
            print()
