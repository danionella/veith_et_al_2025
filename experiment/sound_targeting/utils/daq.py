# Nidaq imports
import nidaqmx
import nidaqmx.system
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogUnscaledReader, DigitalMultiChannelReader, AnalogSingleChannelReader, \
    AnalogMultiChannelReader
from nidaqmx.constants import RegenerationMode, Slope
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, WAIT_INFINITELY
from nidaqmx.utils import flatten_channel_string

import numpy as np

class DAQPlaybackMultiStim:
    def __init__(self, conf, wavsamples, ai_channels=True, accel_channels=False):
        """
        accel_channels: whether to use acceleration channels that supply a dc driving voltage.
        """
        self.conf = conf
        self.wavsamples = wavsamples
        self.sr = self.conf['audio']['rate']
        self.peak_amp = self.conf['audio']['peakAmp']
        self.ai_channels = ai_channels
        self.accel_channels = accel_channels
        self.create_in_task()
        self.create_out_task()

    def create_out_task(self):
        self.taskAO = nidaqmx.Task()
        for ch in self.conf["audio_out"]["chStr"]:
            self.taskAO.ao_channels.add_ao_voltage_chan(ch, max_val=self.peak_amp, min_val=-self.peak_amp)  # wav dim 0
        self.taskAO.timing.cfg_samp_clk_timing(rate=self.sr,
                                               sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                               samps_per_chan=self.wavsamples)
        self.taskAO.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source="/" + self.conf["audio"]["clockStr"] + "/ai/StartTrigger")
        self.writer = AnalogMultiChannelWriter(self.taskAO.out_stream)

    def create_in_task(self):
        self.taskAI = nidaqmx.Task()

        if self.ai_channels:
            for ch in self.conf["audio"]["chStr"]:
                self.taskAI.ai_channels.add_ai_voltage_chan(ch, max_val=self.peak_amp, min_val=-self.peak_amp)
        if self.accel_channels:
            for ch in self.conf["accel"]["chStr"]:
                self.taskAI.ai_channels.add_ai_accel_chan(ch, max_val=self.peak_amp,
                                                          min_val=-self.peak_amp,
                                                          sensitivity=self.conf["accel"]["sensitivity_lookup"][ch],
                                                          sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                                                          current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
                                                          units=nidaqmx.constants.AccelUnits.METERS_PER_SECOND_SQUARED,
                                                          current_excit_val=0.004)
        self.taskAI.timing.cfg_samp_clk_timing(rate=self.sr,
                                               # source='/cDAQ1/ao/SampleClock',
                                               active_edge=nidaqmx.constants.Edge.RISING,
                                               sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                               samps_per_chan=self.wavsamples)
        self.reader = AnalogMultiChannelReader(self.taskAI.in_stream)

    def play_and_record(self, sound):
        data = np.zeros([self.taskAI.number_of_channels, self.wavsamples], 'float64')
        if abs(sound).max() > self.peak_amp:
            sound /= abs(sound).max()
            sound *= self.peak_amp
            print(f"Rescaled sound to be within ouput limits +/-{self.peak_amp}V")
        self.writer.write_many_sample(sound.astype('float64'))
        self.taskAO.start()
        self.taskAI.start()
        self.reader.read_many_sample(data)
        self.taskAI.wait_until_done()
        self.taskAO.wait_until_done()
        self.taskAI.stop()
        self.taskAO.stop()
        self.taskAI.close()
        self.taskAO.close()
        self.create_in_task()
        self.create_out_task()
        return data.T

    def close(self):
        self.taskAI.stop()
        self.taskAO.stop()
        self.taskAI.close()
        self.taskAO.close()
