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
readout.power_sweep(20,0,2,span = 200e3,npts = 201)
readout.fit_power_sweep()
readout.tune_powers()
```

# Finding resonators

# Tuning powers

We want to try scale the tone powers so that they will all have the same non-linearity parameter (so that they are all the same power away from bifurcation). 
It will output normalizing_amplitudes to be used when writing the daq waveform. 

Step 1. Take a power sweep. -You want to take a power sweep. This is an iq sweep at a range of power levels going from reasonably below bifurcation for every resonator to above bifurcation for every resonator.

```
readout.power_sweep(20,0,2) # max_attenuation, min_attenuation, attenuation_step
```

Step 2. Fit the nonlinearity parameter and choose power levels.

Now you want to fit the nonlinearity parameter as a function of power.

```
readout.fit_power_sweep() 
```
Now you want to display the data and correct any bad fit

```
readout.tune_powers()
```


Fitting is done both in magnitude space currently, It will try to set the tone powers so that all of the resonators will all bifurcate at the same power as set by your digital attenuator.

Since the fitting never works perfectly, after the data have been analyzed an interactive plot will pop up. You can scroll through the resonators by pushing the left and right arrows on the keyboard and you can change the plotted power level by pushing the up and down arrows on the keyboard. If you see that the program has chosen the wrong power you can override that choice by either holding shift and right clicking on the bottom plot where you would like the chosen power to be or you can hold shift and press enter to choose the currently displayed power level as the chosen power level. A screenshot of this interface is shown below.


# Taking streaming data

# Doing Polcal
currently, you need to import the polcal class into your python terminal and give it the readout class


```
from detchar import polcal_mkid as polcal
import numpy as np
polcal_sweeper = polcal.PolcalSteppedSweep(readout,fine_span = 300e3,angle_deg_list = np.linspace(0,360,360//5+1),num_lockin_periods = 50) #five degree steps
polcal_sweeper.get_polcal(filename_suffix = "_5_degree_spacing",iq_sweep_all = False)
```

If you need to reinitialize the polcal_sweeper, you have to close the connection to the function generator
```
polcal_sweeper.source.fg.close()
import importlib # if you changed the code otherwise skip this and the next line
importlib.reload(polcal)
polcal_sweeper = PolcalSteppedSweep(readout,fine_span = 300e3,angle_deg_list = np.linsapce(0,360,360//5+1),num_lockin_periods = 50)
```



# To Do list
Gui for configure

print(readout.res_class)

plot show current res_class

choosable non-linearity parameter tune_powers()

instruction on plot for tune_powers()

normalizing_amplitudes to res_class

individual_folder for pol

