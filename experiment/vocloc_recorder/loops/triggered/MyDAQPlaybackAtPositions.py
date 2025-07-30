import time
import numpy as np
import pandas as pd

import nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode, Slope
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import cv2


class MyDAQPlaybackAtPositions:
    def __init__(self, conf):
        """DAQ card based playback with four speakers.
        Using the DAQ card to playback sounds upon trigger event.
        The DAQ's AO is only free if one uses the digital output for camera triggering.
        """
        self.conf = conf
        self.sr = conf["audio"]["rate"]
        self.peak_amp = conf["audio"]["peakAmp"]
        print(f"Peak amplitude is {self.peak_amp} V")
        self.cameraTriggerIn = "/cDAQ1Mod3/PFI1"  # the physical (!) terminal name that corresponds to conf['audio']['chStrTrig']

        # get stimulus set
        from misc.sound_targeting.sound_targeting_field import SoundFieldRecording
        sc = SoundFieldRecording(conf, mkpaths=False)
        print("Start loading stimset...")
        # self.ss, name = sc.get_stimulus_field_set(conf["custom_config"]["fd_stimset"])
        self.ss, name = sc.get_stimulus_field_set_fn(conf["custom_config"]["fn_stimset"])
        print("Ready loading stimset.")
        self.sounds = self.ss['sounds_cdmap0']
        self.soundkeys = list(self.sounds.keys())
        assert self.ss['globalconfig']['audio']['rate'] == self.sr
        del self.ss  # free memory

        self.stimprobs = None  # probabilities for playback
        if "biased_playback_map" in self.conf["custom_config"]:
            # if key of the map is in the stimulus name, assign probability
            biased_playback_map = self.conf["custom_config"]["biased_playback_map"]
            print("Playback of stimuli is biased:")
            print(biased_playback_map)

            self.stimprobs = []
            for x in self.soundkeys:
                unique = 0
                for k, v in biased_playback_map.items():
                    if k in x:
                        unique += 1
                        if unique == 1:
                            self.stimprobs.append(v)
                if unique != 1:
                    print("Assignment of probabilities is not unique!")

            for x, p in zip(self.soundkeys, self.stimprobs):
                print(p, x)
            print("Normalized probabilities: ")
            self.stimprobs = list(np.array(self.stimprobs)/np.array(self.stimprobs).sum())
            for x, p in zip(self.soundkeys, self.stimprobs):
                print(p, x)

        # Light switch
        self.ps_on = conf["custom_config"]["power_switch_on"]
        if self.ps_on:
            self.powerswitch_at_frame = []
            import serial
            self.ser = serial.Serial(conf["custom_config"]["serial_port"], 9600, timeout=1)  # Linux, check for port
            # under Linux:
            # 1. find PORTNAME: dmesg | grep tty
            # 2. give permission: sudo chmod 666 /dev/PORTNAME
            time.sleep(3)
            self.ser.write(b'H')
            time.sleep(0.1)

        keys = list(self.sounds.keys())
        # for key in keys:
        #    if "hann" in key:
        #        self.sounds.pop(key)
        #    if "rubberdrop" in key:
        #        self.sounds.pop(key)

        print("\nSounds in use:")
        print(*list(self.sounds.keys()), sep='\n')
        print("Total Number of Sounds:", len(self.sounds))

        for key, sd in self.sounds.items():
            self.length = sd.shape[1]  # all same length
            self.ny = sd.shape[2]
            self.nx = sd.shape[3]

        # save trigger history
        self.played_name = []
        self.played_at_frame = []
        # self.played_yth_xth = [] # nearest neighbour
        self.played_y_x = []  # pixel position
        self.neighbourlist = []  # bilinear
        self.weightlist = []  # bilinear
        self.min_delay_s = 5  # 5
        self.mean_exp_s = 5  # 15
        self.min_delay_computed = []
        self.count = 0

        self.test = 0
        if self.test:
            self.test_sound = self.my_tone()
            self.test_sound = np.repeat(self.test_sound[None, :, :, :], 4, axis=0)
            self.length = self.test_sound.shape[1]

        # task
        self.create_task()  # uses self.length

        # Show
        self.showgrid = 1
        if self.showgrid:
            im = np.zeros((self.conf["camera"]["height"] // self.conf["videoFunc"]["sub_spatial"],
                           self.conf["camera"]["width"] // self.conf["videoFunc"]["sub_spatial"], 3)).astype('uint8')
            cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Grid', 29 * 30, 15 * 30)
            cv2.imshow('Grid', im)
            cv2.waitKey(1)

        # Skip checks on array shape to reduce writing delays to the nidaq device
        if 1:
            from nidaqmx.constants import FillMode
            def write_many_sample_no_verif(self, data, timeout=10.0):
                """
                Writes one or more floating-point samples to one or more analog
                output channels in a task.

                If the task uses on-demand timing, this method returns only
                after the device generates all samples. On-demand is the default
                timing type if you do not use the timing property on the task to
                configure a sample timing type. If the task uses any timing type
                other than on-demand, this method returns immediately and does
                not wait for the device to generate all samples. Your
                application must determine if the task is done to ensure that
                the device generated all samples.

                Args:
                    data (numpy.ndarray): Contains a 2D NumPy array of
                        floating-point samples to write to the task.

                        Each row corresponds to a channel in the task. Each
                        column corresponds to a sample to write to each channel.
                        The order of the channels in the array corresponds to
                        the order in which you add the channels to the task.
                    timeout (Optional[float]): Specifies the amount of time in
                        seconds to wait for the method to write all samples.
                        NI-DAQmx performs a timeout check only if the method
                        must wait before it writes data. This method returns an
                        error if the time elapses. The default timeout is 10
                        seconds. If you set timeout to
                        nidaqmx.constants.WAIT_INFINITELY, the method waits
                        indefinitely. If you set timeout to 0, the method tries
                        once to write the submitted samples. If the method could
                        not write all the submitted samples, it returns an error
                        and the number of samples successfully written.
                Returns:
                    int:

                    Specifies the actual number of samples this method
                    successfully wrote to each channel in the task.
                """
                # self._verify_array(data, True, True) # commenting this speeds up writing
                auto_start = (
                    self._auto_start if self._auto_start is not nidaqmx.stream_writers.AUTO_START_UNSET else False)
                return self._interpreter.write_analog_f64(self._handle, data.shape[1], auto_start, timeout,
                                                          FillMode.GROUP_BY_CHANNEL.value, data)

            AnalogMultiChannelWriter.write_many_sample = write_many_sample_no_verif

    def get_nn_idx(self, y, x):
        minx = self.minx
        miny = self.miny
        gridw = self.gridw
        gridh = self.gridh
        ny = self.ny
        nx = self.nx

        y -= miny
        x -= minx
        y /= gridh
        x /= gridw
        y = np.clip(y, 0, 1)
        x = np.clip(x, 0, 1)
        y *= (ny - 1)
        x *= (nx - 1)
        yidx = round(y)
        xidx = round(x)
        return yidx, xidx

    def get_nn_and_weights(self, ypos, xpos, exclude_edges=0):
        minx = self.minx
        miny = self.miny
        gridw = self.gridw
        gridh = self.gridh
        ny = self.ny
        nx = self.nx
        dy = gridh / (ny - 1)
        dx = gridw / (nx - 1)
        dy_excl = exclude_edges * dy  # exclude e.g. outer set of stimuli from interpolation
        dx_excl = exclude_edges * dx

        ep = 1e-6  # numerical hack to handle cases when position is outside grid
        # to handle case between two points, factor 1/2 to not revert outside grid correction:
        epy_grid = (ny - 1) * ep / gridh / 2
        epx_grid = (nx - 1) * ep / gridw / 2

        ycl = np.clip(ypos, miny + dy_excl + ep, miny + gridh - dy_excl - ep)
        xcl = np.clip(xpos, minx + dx_excl + ep, minx + gridw - dx_excl - ep)

        # 4 neighbours on grid
        y = ycl - miny
        x = xcl - minx
        y /= gridh
        x /= gridw
        y *= (ny - 1)
        x *= (nx - 1)
        yupp = int(np.ceil(y + epy_grid))  # this way upp and low are never the same
        ylow = int(np.floor(y + epy_grid))
        xupp = int(np.ceil(x + epx_grid))
        xlow = int(np.floor(x + epx_grid))
        neighbours = np.array([[yupp, xupp],  # y1,x1
                               [yupp, xlow],  # y1,x0
                               [ylow, xupp],  # y0,x1
                               [ylow, xlow]  # y0,x0
                               ]).astype(int)

        # Bilinear interpolation in metric space (not grid space)
        ys = np.round(np.linspace(miny, miny + gridh, ny)).astype(int)
        xs = np.round(np.linspace(minx, minx + gridw, nx)).astype(int)
        y1 = ys[yupp]
        y0 = ys[ylow]
        x1 = xs[xupp]
        x0 = xs[xlow]
        area = (y1 - y0) * (x1 - x0)
        weights = np.array([(ycl - y0) * (xcl - x0),
                            (ycl - y0) * (x1 - xcl),
                            (y1 - ycl) * (xcl - x0),
                            (y1 - ycl) * (x1 - xcl)]) / area

        weights[weights < ep] = 0  # num. hack
        weights /= weights.sum()
        return neighbours, weights

    def im_current_grid_nn(self, yidx, xidx, ypos=None, xpos=None):
        im = np.zeros((self.conf["camera"]["height"] // self.conf["videoFunc"]["sub_spatial"],
                       self.conf["camera"]["width"] // self.conf["videoFunc"]["sub_spatial"], 3)).astype('uint8')
        nx = self.nx
        ny = self.ny
        minx = self.minx
        miny = self.miny
        gridw = self.gridw
        gridh = self.gridh
        xs = np.round(np.linspace(minx, minx + gridw, nx)).astype(int)
        ys = np.round(np.linspace(miny, miny + gridh, ny)).astype(int)
        for x in xs:
            for y in ys:
                im = cv2.circle(im, (x, y), 8, (255, 255, 255), -1)
        # picked position
        if yidx is not None:
            im = cv2.circle(im, (xs[xidx], ys[yidx]), 8, (0, 255, 0), -1)
        if ypos is not None:
            im = cv2.circle(im, (xpos, ypos), 3, (255, 0, 255), -1)
        return im

    def im_current_grid_weighted(self, neighbours, weights, ypos=None, xpos=None):
        im = np.zeros((self.conf["camera"]["height"] // self.conf["videoFunc"]["sub_spatial"],
                       self.conf["camera"]["width"] // self.conf["videoFunc"]["sub_spatial"], 3)).astype('uint8')
        nx = self.nx
        ny = self.ny
        minx = self.minx
        miny = self.miny
        gridw = self.gridw
        gridh = self.gridh
        xs = np.round(np.linspace(minx, minx + gridw, nx)).astype(int)
        ys = np.round(np.linspace(miny, miny + gridh, ny)).astype(int)
        for x in xs:
            for y in ys:
                im = cv2.circle(im, (x, y), 8, (255, 255, 255), -1)
        # picked position
        for i, nb in enumerate(neighbours):
            im = cv2.circle(im, (xs[nb[1]], ys[nb[0]]), 8, (0, int(weights[i] * 255), 0), -1)
        if ypos is not None:
            im = cv2.circle(im, (xpos, ypos), 3, (255, 0, 255), -1)
        return im

    def upon_trigger(self, vCamFrame, q_video2triggered):
        """
        """
        # Get position
        content = q_video2triggered.get()

        # tstart = time.time()
        # print(f"Triggered_func: {content}")
        x, y, self.minx, self.miny, self.gridw, self.gridh = content
        q_video2triggered.clear()

        # nearest neighbour
        # yidx, xidx = self.get_nn_idx(y, x)
        # print(xidx, yidx)

        # bilinear interpolation
        nb, weights = self.get_nn_and_weights(y, x, exclude_edges=1)
        # print(nb, weights)
        # tstop = time.time()
        # print(f"Time delay {tstop - tstart}")
        # tstart = tstop

        # Sound
        if self.stimprobs is not None:
            name = np.random.choice(self.soundkeys, p=self.stimprobs)
        else:
            name = np.random.choice(self.soundkeys)

        sd = self.sounds[name]
        self.played_name.append(name)
        self.neighbourlist.append(nb)  # bilinear
        self.weightlist.append(weights)  # bilinear
        # self.played_yth_xth.append([yidx, xidx]) #nearest neighbour
        self.played_y_x.append([y, x])  # pixel position
        # Play
        print(f"{self.count}: {name[-50:]}")
        if not self.test:
            # sdtmp = sd[:, :, yidx, xidx].astype('float64') #nearest neighbour
            sdtmp = np.sum(np.array([w * sd[:, :, yidx, xidx] for w, (yidx, xidx) in zip(weights, nb)]), axis=0).astype(
                'float64')  # bilinear interpol
        else:
            sdtmp = 0.1 * self.test_sound[:, :, yidx, xidx].astype('float64')
        # tstop = time.time()
        # print(f"Time delay {tstop - tstart}")
        # tstart = tstop

        print(abs(sdtmp).max())
        self.writer.write_many_sample(sdtmp)
        # tstop = time.time()
        # print(f"Time delay {tstop - tstart}")
        # tstart = tstop
        # Delays
        # time_before = 0  # without delay, the sound would always be played when the fish is at the active zone's edge
        # time.sleep(time_before)

        if self.ps_on:
            self.powerswitch_at_frame.append(vCamFrame.value)
            self.ser.write(b'L')
            delay_rounded = int(self.conf["custom_config"]["power_switch_delay"] * self.conf['trigger']['rate']) / \
                            self.conf['trigger']['rate']
            time.sleep(delay_rounded)

        self.played_at_frame.append(vCamFrame.value)  # value is taken at this time

        # tstop = time.time()
        # print(f"Time delay {tstop - tstart}")
        # start = tstop
        self.taskAO.start()
        self.taskAO.wait_until_done()  # reserves task for some time
        self.taskAO.stop()
        self.taskAO.close()
        self.create_task()
        self.count += 1

        # wait minimum time after end of stimulus
        delay = self.min_delay_s + np.random.exponential(scale=self.mean_exp_s)
        print(f"wait {delay:.2f}s")
        self.min_delay_computed.append(delay)

        # Show activation grid position
        if self.showgrid:
            # im = self.im_current_grid_nn(yidx, xidx) #nearest neighbour
            img = self.im_current_grid_weighted(nb, weights, ypos=y, xpos=x)  # bilinear
            cv2.imshow('Grid', img.astype('uint8'))
            cv2.waitKey(100)  # doesn't block the thread, just make sure the image is shown

        if self.ps_on:
            delay -= 1
            time.sleep(1)
            self.ser.write(b'H')

        time.sleep(delay)
        return

    def close(self):
        if self.ps_on:
            playback_triggers = pd.DataFrame({'played_file': self.played_name,
                                              'played_at_frame': self.played_at_frame,
                                              'powerswitch_at_frame': self.powerswitch_at_frame,
                                              # 'played_yth_xth': self.played_yth_xth,# nearest neighbour,
                                              'played_y_x': self.played_y_x,
                                              'neighbourlist': self.neighbourlist,  # bilinear
                                              'weightlist': self.weightlist,  # bilinear
                                              'min_delay_s': self.min_delay_s,
                                              'mean_exp_s': self.mean_exp_s,
                                              'min_delay_computed': self.min_delay_computed})
        else:
            playback_triggers = pd.DataFrame({'played_file': self.played_name,
                                              'played_at_frame': self.played_at_frame,
                                              # 'played_yth_xth': self.played_yth_xth,# nearest neighbour,
                                              'played_y_x': self.played_y_x,
                                              'neighbourlist': self.neighbourlist,  # bilinear
                                              'weightlist': self.weightlist,  # bilinear
                                              'min_delay_s': self.min_delay_s,
                                              'mean_exp_s': self.mean_exp_s,
                                              'min_delay_computed': self.min_delay_computed})
        print(playback_triggers)

        fn = self.conf['pSavBaseName'] + 'triggeredPlayback.csv'
        playback_triggers.to_csv(fn)
        self.taskAO.stop()
        self.taskAO.close()
        if self.showgrid:
            cv2.destroyAllWindows()

    def create_task(self):
        self.taskAO = nidaqmx.Task()
        for ch in self.conf["audio_out"]["chStr"]:
            self.taskAO.ao_channels.add_ao_voltage_chan(ch, max_val=self.peak_amp, min_val=-self.peak_amp)
        self.taskAO.timing.cfg_samp_clk_timing(rate=self.sr,
                                               sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                               samps_per_chan=self.length)
        self.taskAO.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=self.cameraTriggerIn)
        self.writer = AnalogMultiChannelWriter(self.taskAO.out_stream)

    def my_tone(self):
        def smoothing(nsamples, frac=0.4):
            """
            returns an inverted sigmoid of nsamples length that starts at 1 and stops at 0
            :param nsamples: length of the sigmoid
            :param frac: standard deviation of the sigmoid as sample fraction
            """
            nsamples = int(nsamples)
            x = np.linspace(-nsamples, nsamples, nsamples * 2)
            m = 0
            a = 0.1
            s = frac * nsamples
            # gauss = lambda x, m, s: np.exp(-((x - m) / s) ** 2)
            sigmoid = lambda x: np.exp(x * a) / (1 + np.exp(x * a))
            smooth = sigmoid(x)[nsamples:]
            smooth -= smooth[-1]  # set last value to 0
            smooth /= smooth[0]  # normalize
            return smooth

        samplerate = 51200
        a = np.linspace(0, 0.5, 7)
        f2 = np.array([256 * (2) ** (x / 12) for x in range(8, 15)])
        noise = 0.5 * np.random.rand(samplerate)
        # f = np.linspace(f_min, f_max, 7)
        f = np.array([256 * (2) ** (x / 12) for x in range(1, 8)])
        t = np.arange(0, samplerate) / samplerate
        sounds = np.zeros((samplerate, 7, 7))
        print(sounds.shape)
        ramp_duration = 0.001
        ramp_samples = int(ramp_duration * samplerate)
        smooth = smoothing(ramp_samples)
        for i, freq2 in enumerate(f2):
            for j, freq in enumerate(f):
                # sin = amp * noise + 0.5 * np.sin(2 * np.pi * freq * t)
                sin = 0.5 * np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
                sin[:ramp_samples] *= smooth[::-1]
                sin[-ramp_samples:] *= smooth
                sounds[:, i, j] = sin
        return sounds