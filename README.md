# KID_multitone_readout

Getting started

first cd into the KID_multitone_readout directory

then start python by typing 
```
python

import multitone_readout as mr
readout = mr.readout()
help(readout)
Press Q to exit
readout.vna_sweep(400e6,900e6)
readout.find_resonators()
readout.res_class # to inspect res_class
readout.save_res_class()
readout.mask_collisions(1e6)
readout.save_res_class()
readout.write_resonator_tones()
readout.iq_sweep(span = 200e3)
# if the power is too high turn donw input attenuator
readout.input_attenuator.set_attenuation(10)
readout.iq_sweep()
readout.retune_resonators(find_min = True)
readout.iq_sweep()
readout.power_sweep(20,0,2)
readout.fit_power_sweep()
readout.tune_powers()
```

# Tuning powers

This function tries to scale the tone powers so that they will all have the same non-linearity parameter (so that they are all the same power away from bifurcation). 
It will output normalizing_amplitudes to be used when writing the daq waveform. 



To Do list
Gui for configure

print(readout.res_class)

plot show current res_class

choosable non-linearity parameter tune_powers()

instruction on plot for tune_powers()

