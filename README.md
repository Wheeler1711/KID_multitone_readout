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
readout.vna_sweep(400e6,1200e6, average_time = 0.001, min_tone_spacing = 20e6) # Varies depending on the MKID array (power is distributed between tones)
readout.input_attenuator.set_attenuation(0); # May vary depending on the setup, in the first time the input attenuator was bypassed  
readout.output_attenuation.set_attenuation(25); # May vary depending on the setup 
readout.find_resonators() # Select smoothing for the baseline removal and press Q (opens a new window)
readout.res_class # to inspect res_class
readout.save_res_class()
readout.mask_collisions(1e6)
readout.save_res_class()
readout.write_resonator_tones()
readout.iq_sweep(span = 200e3)
# if the power is too high turn down input attenuator
readout.input_attenuator.set_attenuation(10)
readout.iq_sweep()
readout.retune_resonators(find_min = True)
readout.iq_sweep()
readout.power_sweep(20,0,2,span = 200e3,npts = 201) # This does not work if input attenuator is not used 
readout.dac_power_sweep(20, 0, 2) # Directly scaling the FPGA generated waveform # Use this, input and output powers should match to obtain the best S21
readout.fit_power_sweep()
readout.tune_powers()
```


If you need to restart the code you can get back to the same setting using the following
```
import multitone_readout as mr
readout = mr.readout()
readout.load_res_class()
readout.normalizing_amplitudes = np.load("normalizing_amplitudes.npy")
readout.write_resonator_tones()
readout.iq_sweep()
readout.retune_resonators(find_min = True)
```

If you want to clear the resonator list and start over
```
readout.res_class = resonator_class.resonators_class([])
```

If you want to load a previous list of resonators 
```
readout.res_class = np.load("YYMMDD_HHMMSS_resonators.csv");
```

If you want to change the save directory after reloading and adjusting the list of resonators 
```
print(readout.data_dir) # shows the existing save directory
readout.data_dir = "/place_the_new_path_here/"
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
Monitor/Take stream data 
```
readout.monitor(); # If you want to see roughly what's going on
readout.take_noise_data(downsampling = 1); # Saves the downsampled data. When downsampling = 1, the sample rate is 250 kS/s
```

# Polcal - rotation only 
The alignment of the source is very important! If the source is on the XY stage (even if you would not do beam mapping) please set the XY stage first to home and reset it to (0, 0). It is very easy to loose the coordinates. 

Currently, you need to import the polcal class into your python terminal and give it the readout class

```
from detchar import polcal_mkid as polcal
import numpy as np
polcal_sweeper = polcal.PolcalSteppedSweep(readout,fine_span = 300e3,angle_deg_list = np.linspace(0,360,360//5+1),num_lockin_periods = 50) #five degree steps
polcal_sweeper.get_polcal(filename_suffix = "_5_degree_spacing",iq_sweep_all = False, downsampling=250) # Downsampling is a new feature
# Do not change this directory name afterwards, calibration routine may break 
```

If you need to reinitialize the polcal_sweeper, you have to close the connection to the function generator
```
polcal_sweeper.source.fg.close()
import importlib # if you changed the code otherwise skip this and the next line
importlib.reload(polcal)
polcal_sweeper = polcal.PolcalSteppedSweep(readout,fine_span = 300e3,angle_deg_list = np.linspace(0,360,360//5+1),num_lockin_periods = 50)
# Do not change this directory name afterwards, calibration routine may break 
```

# Polcal - rotation and XY mapping 
The alignment of the source is very important! Please set home_xy =True to make sure you don't loose the coordinates. 
```
from detchar import polcal_mkid as polcal
xy_list = polcal.make_xy_list(340, 350, 11, 5) # (x_center (mm), y_center (mm), npts, step (mm))
grid_angle = 0; # Set the polarization angle (deg)
polcal_sweeper = polcal.BeamMapSingleGridAngle(readout, xy_list, grid_angle, filename_suffix = "_xy_0deg_5mm_step", home_xy = True, num_lockin_periods = 50, wait_s = 1) # Default wait_s = 0.1 seconds
# Do not change this directory name afterwards, calibration routine may break 
polcal_sweeper.acquire(downsampling=250) # Downsampling is a new feature
```
If you want to move the xy-stage in absolute coordinates (mm) referenced to "home" (0, 0)
```
polcal_sweeper.xy.move_absolute(340, 350, 5, 5);
```

# Calibrate and average polcal data 
```
# Multitone readout measures 250 kS/s
# Calibration routine reduces it to 1 kS/s
# Folder needs to be inerted manually in the code 
#
python calibrate_polcal.py 
```

# Enable TTL and check inputs and outputs
```
pktsrcsel.py udp://10.0.15.11 timestamp=ENABLED                    # Enables timestamp 
eidtest.py --ctrl=udp://10.0.15.11 -d | grep -e riseen -e fallen   # Gives the IDs for fallng and rising edge trigger signals
# fallen = 2147483648
# riseen = 3221225472
regrw.py udp://10.0.15.11 0x06000038=0xffffffff 0x0600003c=0xffffffff
```

# Reset rfsoc in linux (for example after power outage) 
```
busybox telnet 192.168.4.11            # log in to Raspberry pi
  login: root
  echo 0 > /sys/class/gpio/gpio5/value 
  echo 1 > /sys/class/gpio/gpio5/value # Reset Raspberry pi remotely 
busybox telnet 192.168.6.11            # log in to rfsoc (not necessary, Raspberry pi controls rfsoc)

# Start-up commands for FPGA and Raspberry PI 
regrw.py udp://10.0.15.11 0x08000018=0x6a777787 0x08000048=0x5154 0x08000050=0xf25af2a5  0x09000018=0x67777701
ncoconf.py udp://10.0.15.11 .DAC=896e6 .ADC=-896e6
busybox telnet 192.168.4.11 3021       # look at rfsoc in linux console
ping 192.168.6.11                      # Check connection


ifconfig eno2:1 192.168.30.21/24 up    # Connect XY stage via Ethernet 
ping 192.168.30.100                    # Check connection

busybox telnet 192.168.0.231           # Remote connect to power connectors
ps 1                                   # Set all ports to 1 or 0 (ON or OFF)
pset 5 1                               # Set one port to 1 or 0
pset 6 1
pset 17 1                              # etc.
logout



```
# Remote view of Linux using RealVNC Viewer (Windows Powershell)
```
vncserver -geometry 1800x1000 :3
```


# Move large folders from Linux to Windows
```
# ----- On Linux ----- 

# list all files with '*cal.p' extension in a folder
ls -l folder_name

# Check free space available in tmp folder 
df -h /tmp/ 

# Create a zip file to temporary /tmp/ folder (all files with '*cal.p' extension) 
zip -r /tmp/folder_name.zip folder_name/*_cal.p

# Check the size of the created zip file 
du -ms /tmp/folder_name.zip 

# ----- On Windows Powershell ----- 
# Connect SSH
ssh -l pcuser 687hawc

# Go to the folder where you want to copy the .zip folder 
PS Z:> scp pcuser@687hawc:/tmp/folder_name.zip ./ 


# ----- On Linux ----- 
# Check how much space available in /tmp/ folder
df -h /tmp/

# /tmp/ folder is not large, you might need to remove folders at some point 
rm -vi /tmp/folder_name.zip 
```

# Useful general commands 
```
df -T                        # shows available and used memory, FPGA output 250 kS/s, so space can run out
nmcli device show            # lists all devices connected
df /                         # returns the recycling bin location, in our case ~./dev/nvme0n1p2
sudo debugfs dev/nvme0n1p2   # opens debugfs 
lsdel                        # lists deleted files
q                            # exits debugfs
cp xxx.txt /tmp/xxx.txt      # copy to a temporary folder 
```

# To Do list
Gui for configure

print(readout.res_class)

plot show current res_class

choosable non-linearity parameter tune_powers()

instruction on plot for tune_powers()

normalizing_amplitudes to res_class

in polcal.BeamMapSingleGridAngle(): home_xy == True -> Setting leaves the code hanging sometimes 


