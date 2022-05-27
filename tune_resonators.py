import numpy as np
import matplotlib.pyplot as plt
import os

'''
script for retuning the kids either to the minimum of the resonance
or to the max seperation in the iq loop (best place for streaming noise)
modified from https://github.com/sbg2133/kidPy/
'''


def find_max_didq(z, look_around):
    Is = np.real(z)
    Qs = np.imag(z)
    pos_offset_I = np.roll(Is, 1, axis=0)
    neg_offset_I = np.roll(Is, -1, axis=0)
    pos_offset_Q = np.roll(Qs, 1, axis=0)
    neg_offset_Q = np.roll(Qs, -1, axis=0)
    pos_dist = np.sqrt((Is-pos_offset_I)**2+(Qs-pos_offset_Q)**2)
    neg_dist = np.sqrt((Is-neg_offset_I)**2+(Qs-neg_offset_Q)**2)
    ave_dist = (pos_dist + neg_dist)/2.
    # zero out the last and first values
    ave_dist[0, :] = 0
    ave_dist[ave_dist.shape[0]-1, :] = 0
    min_index = np.argmax(ave_dist[Is.shape[0]//2-look_around:Is.shape[0]//2+look_around],
                          axis=0)+(Is.shape[0]//2-look_around)
    return min_index


class interactive_plot(object):

    def __init__(self, chan_freqs, z, look_around=2,stream_data = None, retune=True, find_min=True):
        self.find_min = find_min
        self.retune = retune
        self.Is = np.real(z)
        self.Qs = np.imag(z)
        self.z = z
        self.stream_data = stream_data
        self.chan_freqs = chan_freqs
        self.targ_size = chan_freqs.shape[0]
        self.look_around = look_around
        self.plot_index = 0
        self.res_index_overide = np.asarray((), dtype=np.int16)
        self.overide_freq_index = np.asarray((), dtype=np.int16)
        self.shift_is_held = False
        if self.find_min:
            self.min_index = np.argmin(self.Is[self.targ_size//2-self.look_around:self.targ_size//2+self.look_around]**2+self.
                                       Qs[self.targ_size//2-self.look_around:self.targ_size//2+look_around]**2, axis=0) +\
                                       (self.targ_size//2-look_around)
        else:
            self.min_index = find_max_didq(self.z, self.look_around)
        self.fig = plt.figure(1, figsize=(13, 6))
        self.ax = self.fig.add_subplot(121)
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_xlabel("Frequecy (MHz)")
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_ylabel("Q")
        self.ax2.set_xlabel("I")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        if self.stream_data is not None:
            self.s2, = self.ax2.plot(np.real(self.stream_data[:, self.plot_index]),
                                     np.imag(self.stream_data[:, self.plot_index]), '.')
        self.l1, = self.ax.plot(self.chan_freqs[:, self.plot_index]/10**6, 10*np.log10(
            self.Is[:, self.plot_index]**2+self.Qs[:, self.plot_index]**2), '-o',mec = "k")
        self.l2, = self.ax2.plot(
            self.Is[:, self.plot_index], self.Qs[:, self.plot_index], '-o',mec = "k")
        if self.retune:
            self.p1, = self.ax.plot(self.chan_freqs[self.min_index[self.plot_index], self.plot_index]/10**6,
                                    10*np.log10(self.Is[self.min_index[self.plot_index],self.plot_index]**2+\
                                                self.Qs[self.min_index[self.plot_index], self.plot_index]**2), '*', markersize=15)
            self.p2, = self.ax2.plot(self.Is[self.min_index[self.plot_index], self.plot_index],
                                     self.Qs[self.min_index[self.plot_index], self.plot_index], '*', markersize=15)
            
        self.ax.set_title("Resonator Index "+str(self.plot_index))
        if self.retune:
            self.ax2.set_title("Look Around Points "+str(self.look_around))
        print("")
        print("Interactive Resonance Tuning Activated")
        print("Use left and right arrows to switch between resonators")
        if retune:
            print("Use the up and down arrows to change look around points")
            print("Hold shift and right click on the magnitude plot to overide tone position")
        plt.show(block=True)

    def refresh_plot(self):
        self.l1.set_data(self.chan_freqs[:, self.plot_index]/10**6, 10*np.log10(
            self.Is[:, self.plot_index]**2+self.Qs[:, self.plot_index]**2))
        if self.retune:
            if (self.res_index_overide == self.plot_index).any():
                index = np.argwhere(self.res_index_overide ==
                                    self.plot_index)[0][0]
                #print("index is",index)
                self.p1.set_data(self.chan_freqs[self.overide_freq_index[index], self.plot_index]/10**6,
                                 10*np.log10(self.Is[self.overide_freq_index[index], self.plot_index]**2 +
                                             self.Qs[self.overide_freq_index[index], self.plot_index]**2))
            else:
                self.p1.set_data(self.chan_freqs[self.min_index[self.plot_index], self.plot_index]/10**6,
                                 10*np.log10(self.Is[self.min_index[self.plot_index], self.plot_index]**2 +
                                             self.Qs[self.min_index[self.plot_index], self.plot_index]**2))
                    
        self.ax.relim()
        self.ax.autoscale()
        self.ax.set_title("Resonator Index "+str(self.plot_index))
        if self.retune:
            self.ax2.set_title("Look Around Points "+str(self.look_around))
        self.l2.set_data((self.Is[:, self.plot_index],
                          self.Qs[:, self.plot_index]))
        if self.retune:
            if (self.res_index_overide == self.plot_index).any():
                self.p2.set_data(self.Is[self.overide_freq_index[index], self.plot_index],
                                 self.Qs[self.overide_freq_index[index], self.plot_index])
            else:
                self.p2.set_data(self.Is[self.min_index[self.plot_index], self.plot_index],
                                 self.Qs[self.min_index[self.plot_index], self.plot_index])

        if self.stream_data is not None:
            self.s2.set_data(np.real(self.stream_data[:, self.plot_index]),
                                 np.imag(self.stream_data[:, self.plot_index]))
        self.ax2.relim()
        self.ax2.autoscale()
        plt.draw()

    def on_key_press(self, event):
        # print event.key
        if event.key == 'right':
            if self.plot_index != self.chan_freqs.shape[1]-1:
                self.plot_index = self.plot_index + 1
                self.refresh_plot()

        if event.key == 'left':
            if self.plot_index != 0:
                self.plot_index = self.plot_index - 1
                self.refresh_plot()

        if event.key == 'up':
            if self.look_around != self.chan_freqs.shape[0]//2:
                self.look_around = self.look_around + 1
                if self.find_min:
                    self.min_index = np.argmin(self.Is[self.targ_size//2-self.look_around:self.targ_size//2+self.look_around]**2 +
                                               self.Qs[self.targ_size//2-self.look_around:self.targ_size//2+self.look_around]**2,
                                               axis=0)+(self.targ_size//2-self.look_around)
                else:
                    self.min_index = find_max_didq(self.z, self.look_around)
                self.refresh_plot()

        if event.key == 'down':
            if self.look_around != 1:
                self.look_around = self.look_around - 1
                if self.find_min:
                    self.min_index = np.argmin(self.Is[self.targ_size//2-self.look_around:self.targ_size//2+self.look_around]**2 +
                                               self.Qs[self.targ_size//2-self.look_around:self.targ_size//2+self.look_around]**2,
                                               axis=0)+(self.targ_size//2-self.look_around)
                else:
                    self.min_index = find_max_didq(self.z, self.look_around)
                self.refresh_plot()

        if event.key == 'shift':
            self.shift_is_held = True
            #print("shift pressed")

    def on_key_release(self, event):
        if event.key == "shift":
            self.shift_is_held = False
            #print("shift released")

    def onClick(self, event):
        if event.button == 3:
            if self.shift_is_held:
                print("overiding point selection", event.xdata)
                # print(self.chan_freqs[:,self.plot_index][50])
                #print((self.res_index_overide == self.plot_index).any())
                if (self.res_index_overide == self.plot_index).any():
                    replace_index = np.argwhere(
                        self.res_index_overide == self.plot_index)[0][0]
                    new_freq = np.argmin(
                        np.abs(event.xdata-self.chan_freqs[:, self.plot_index]/10**6))
                    self.overide_freq_index[replace_index] = np.int(new_freq)

                else:
                    self.res_index_overide = np.append(
                        self.res_index_overide, np.int(np.asarray(self.plot_index)))
                    # print(self.res_index_overide)
                    new_freq = np.argmin(
                        np.abs(event.xdata-self.chan_freqs[:, self.plot_index]/10**6))
                    #print("new index is ",new_freq)
                    self.overide_freq_index = np.append(
                        self.overide_freq_index, np.int(np.asarray(new_freq)))
                    # print(self.overide_freq_index)

                self.refresh_plot()


def tune_kids(f, z, find_min=True, interactive=True, **kwargs):
    # f and z should have shape (npts_sweep,n_res)
    #iq_dict = read_iq_sweep(filename)
    if "look_around" in kwargs:
        print("you are using " +
              str(kwargs['look_around'])+" look around points")
        look_around = kwargs['look_around']
    else:
        look_around = f.shape[0]//2
    if find_min:  # fine the minimum
        print("centering on minimum")
        if interactive:
            ip = interactive_plot(f, z, look_around)
            for i in range(0, len(ip.res_index_overide)):
                ip.min_index[ip.res_index_overide[i]
                             ] = ip.overide_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = np.argmin(np.abs(z)**2, axis=0)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    else:  # find the max of dIdQ
        print("centering on max dIdQ")
        if interactive:
            ip = interactive_plot(f, z, look_around, find_min=False)
            for i in range(0, len(ip.res_index_overide)):
                ip.min_index[ip.res_index_overide[i]
                             ] = ip.overide_freq_index[i]
            new_freqs = f[(ip.min_index, np.arange(0, f.shape[1]))]
        else:
            min_index = find_max_didq(z, look_around)
            new_freqs = f[(min_index, np.arange(0, f.shape[1]))]
    return new_freqs
