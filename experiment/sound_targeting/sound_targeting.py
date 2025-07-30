import sys
import os
import shutil
import numpy as np
import scipy as sc
from scipy import signal
import h5py as h5
import pandas as pd
import time
import scipy.signal as sig
from scipy.signal import butter, lfilter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.mlab import psd, specgram
import datetime
from collections import OrderedDict
import cv2
import soundfile as sf

# Nidaq imports
import nidaqmx
import nidaqmx.system
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter
from nidaqmx.stream_readers import AnalogUnscaledReader, DigitalMultiChannelReader, AnalogSingleChannelReader, \
    AnalogMultiChannelReader
from nidaqmx.constants import RegenerationMode, Slope
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, WAIT_INFINITELY
from nidaqmx.utils import flatten_channel_string

# ---------------------------------------------
# DAQ CONFIG
conf = {
    "audio_in": {
        "rate": 51200.0,
        "clockStr": "cDAQ1",
        "chStr": ["cDAQ1Mod1/ai0"],  # "cDAQ1Mod1/ai3", "cDAQ1Mod1/ai4"
        "chStrTrig": "cDAQ1Mod3/port0/line1",
        "peakAmp": 4, # 1.1
        "szChunk": 100000.0,
        "audioDeviceName": "Microsoft Soundmapper - Output"
    },
    "accel_in": {
        "rate": 51200.0,
        "clockStr": "cDAQ1",
        "chStr": ["cDAQ1Mod1/ai3", "cDAQ1Mod1/ai4", "cDAQ1Mod1/ai5"],
        "sensitivity_lookup": {"cDAQ1Mod1/ai3": 98.2, "cDAQ1Mod1/ai4": 99.0, "cDAQ1Mod1/ai5": 103.3},
        # mV/g as per calibration sheet
        "chStrTrig": "cDAQ1Mod3/port0/line1",
        "peakAmp": 1.1,
        "szChunk": 100000.0,
        "audioDeviceName": "Microsoft Soundmapper - Output"
    },
    "audio_out": {
        "chStr": ["cDAQ1Mod2/ao0", "cDAQ1Mod2/ao1", "cDAQ1Mod2/ao2", "cDAQ1Mod2/ao3"]
    }
}





# ---------------------------------------------
# PARAMS
params = dict()
# conversion of measurements to pressure
# turn measurement into volts, see manual NI 9231 for unscaled recordings, 24 bit
meas_2_vDAQ = 1 # 2 ** 23 * (610715 * 10 ** -12)  # ~5.123
# volts at AS-1 hydrophone, 50dB gain of inverted single line PA4 amplifier
vDAQ_2_vAS1 = 1 / 10 ** (50 / 20)  # todo, 20 or 10?
# pressure from AS-1 sensitivity 40uV / Pascal
vAS1_2_p = 1 / (40e-6)
# all at once
m2p = meas_2_vDAQ * vDAQ_2_vAS1 * vAS1_2_p
params["m2p"] = m2p

# medium
params["rho"] = 1000.0
params["c"] = 1500.0


# ---------------------------------------------
# OBSERVABLES (all in SI UNITS)
def get_pressure(data):
    """
    define what hydrophone measures the relevant pressure
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: pressure
    """
    ch = 0

    p = m2p * data[:, ch]
    return p


def get_acceleration_x(data):
    """
    define which hydrophones measure the relevant acceleration
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: acceleration
    """
    ch_pair = [1, 2]
    distance = 0.065
    params["acc_distance"] = distance

    a = - m2p * (data[:, ch_pair[1]] - data[:, ch_pair[0]]) / distance / params["rho"]  # todo check sign
    return a

def get_acceleration_y(data):
    """
    define which hydrophones measure the relevant acceleration
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: acceleration
    """
    ch_pair = [3, 4]
    distance = 0.065
    params["acc_distance"] = distance

    a = - m2p * (data[:, ch_pair[1]] - data[:, ch_pair[0]]) / distance / params["rho"]  # todo check sign
    return a
def get_pcb_ax(data):
    """
    define which hydrophones measure the relevant acceleration
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: acceleration
    """
    return data[:, 3]


def get_pcb_ay(data):
    """
    define which hydrophones measure the relevant acceleration
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: acceleration
    """
    return data[:, 4]


def get_pcb_az(data):
    """
    define which hydrophones measure the relevant acceleration
    :param data: recording from all input hydrophones in order of ["audio_in"]["chStr"]
    :return: acceleration
    """
    return data[:, 5]


def get_pressure_mean(data):
    p = m2p * np.mean(data, axis=1)
    return p





# ---------------------------------------------
# SPEAKERS
def get_first_speakers(n=None):
    """
    Default speakers for playback: take first n speakers, given n signals to playback.
    :param n: number of speakers to take, all speakers if None.
    :return: speaker configuration
    """
    nch = len(conf["audio_out"]["chStr"])
    speakers = np.identity(nch, dtype=bool)[:n].tolist()
    return speakers


def get_speakers_custom():
    """
    Define manual speaker configuration. Given n observables, one can target n signals.
    A signal can be mapped to a combination of speakers.
    """
    speakers = [[1, 1, 0], [0, 1, 0]]  # n x number of connected speakers
    return speakers


# ---------------------------------------------
# OUTPUT
now = datetime.datetime.now()
nowstr = now.strftime('%Y-%m-%d_%H-%M-%S')
pth = r'{}_stimset'.format(nowstr)
if os.path.exists(pth):
    shutil.rmtree(pth)
os.makedirs(pth)

plotpth = pth


# ---------------------------------------------
def main():
    align_speakers()
    #play_simple_function()
    #rec_simple_function(repeats=30,speaker_choices=[[1,0,0,0]])
    #rec_simple_function_across_amps(repeats=20, speaker_choices=[[1, 0, 0, 0]], amplitudes=[0.025,0.3,0.6,1,2,2.5,3,3.5,4,5])

    # make_pressure_inversions()
    # make_stimulus_set()
    # make_stimulus_set_2()

    # run()
    # signal_sweep()


def align_speakers():

    obs = {"pressure": get_pressure,
           "acceleration_x-dp": get_acceleration_x,
           "acceleration_x-pcb": get_pcb_ax,
           "acceleration_y-pcb": get_pcb_ay,
           "acceleration_z-pcb": get_pcb_az}

    obs = {"pressureMean": get_pressure_mean,
           "acceleration_x": get_acceleration_x,
           "acceleration_y": get_acceleration_y}

    obs = {"pressure": get_pressure}
    # sine
    wav = .8 * ramped_sinusoid(conf["audio_in"]["rate"], freq=900, duration=0.05)

    # delta pulse
    pp = {"duration": 0.03, "offset": 0.005}
    dp = deltapulse(samplerate=conf["audio_in"]["rate"], **pp)
    wav = dp / np.max(abs(dp))

    # gaussian pulse
    #wav = gaussian_double_pulse(samplerate=conf["audio_in"]["rate"], center_frequency=800, pulse_duration=0.05)

    # custom sound
    #wav, sr = sf.read("/home/jlab-pc09/Documents/data/playback/wav_folder/rubberdrop_HP100LP20000_1stDrop_1stpart_normalized_smalldelay_shortenedForDAQ_upsampled51200.wav")
    #assert sr == conf["audio_in"]["rate"]

    # for i, freq in enumerate(np.linspace(500,1000,14)):
    for i in range(100):
        # wav = ramped_sinusoid(conf["audio_in"]["rate"], freq=freq, duration=0.05)
        print("single")
        recs = rec_repeated(pth=pth, handle=str(0), base_wav=wav, sp=[1, 0, 0, 0], repeat=5, info=None, save=False)
        plot_repeated(pth=pth, handle=str(0), obs=obs, plot_each=False, show=True, recs=recs, show_fft_coefs=True, fft_cutoff=5000)
        # print("three")
        # recs = rec_repeated(pth=pth, handle=str(0), base_wav=wav, sp=[1, 0, -1, -1], repeat=1, info=None, save=False)
        # plot_repeated(pth=pth, handle=str(0), obs=obs, plot_each=False, show=True, recs=recs)


def record_rubberdrop_sounds():
    obs = {"pressureMean": get_pressure_mean,
           "pressureCenter": get_pressure,
           "acceleration_x": get_acceleration_x,
           "acceleration_y": get_acceleration_y}
    sr = conf["audio_in"]["rate"]
    # loudspeaker in air to trigger human to release the item
    wav = 0.001 * ramped_sinusoid(conf["audio_in"]["rate"], freq=900, duration=.8)
    wav[int(sr * 0.3):] = 0
    info = {}
    info["item"] = 'rubber'
    info["side"] = 'left'
    info["distance_to_center"] = 0.2
    info["height"] = 0.1
    j = 0
    for i in range(10):
        print("DROP!", j)
        time.sleep(5)
        print("now")
        recs = rec_repeated(pth=pth, handle=f"rubber_{j}", base_wav=wav, sp=[1, 0, 0, 0], repeat=1, info=info,
                            save=True)
        plot_repeated(pth=pth, handle=f"rubber_{j}", obs=obs, plot_each=False, show=True, recs=recs)

        if input("Good?") == str(1):
            j += 1
            print("Keep it.")


def make_stimulus_set():
    """
    Creates rubberdrops and their pressure-inverted versions, as well as gammatone sounds. Used in 202105DireHearingCL_trick_gamma
    :return:
    """

    sr = conf["audio_in"]["rate"]
    obs = {"pressureCenter": get_pressure,
           "acceleration_y": get_acceleration_y,
           "acceleration_x": get_acceleration_x}

    stimset = {}
    repeat = 10

    # record impulse responses for each hydrophone

    print("record impulse responses.")
    rec_irs(pth, repeat=repeat)
    plot_irs(irpth=pth, plot_each=False, obs=obs)

    # make kernels for observables
    print("make kernels.")
    kernels = make_kernels(irpth=pth, obs=obs)
    lenkernel = np.array(kernels["pressureCenter"]).shape[1]  # targets are shifted by length of kernel, don't crop them
    plot_kernels(kpth=pth)

    # crop stimuli

    croptime = int(sr * 0.018)
    print("make rubber drop sounds")
    # Make rubberdrop, correc amps
    rubberdrops, rparams = make_all_rubberdrop_sounds(pth, repeat)
    for key, val in rubberdrops.items():
        rubberdrops[key]["sound"] = val["sound"][:, croptime:-croptime]
        if "target" in rubberdrops[key].keys():
            for k, v in rubberdrops[key]["target"].items():
                rubberdrops[key]["target"][k] = v[croptime - lenkernel:-croptime]
    length = rubberdrops[key]["sound"].shape[1]

    croptime = int(sr * 0.0205)
    print("make gammatone sounds")
    # Gammatone sounds (targeted)
    gammatones, gparams = make_all_gammatone_sounds(pth, repeat)
    for key, val in gammatones.items():
        gammatones[key]["sound"] = val["sound"][:, croptime:croptime + length]
        for k, v in gammatones[key]["target"].items():
            gammatones[key]["target"][k] = v[croptime - lenkernel:croptime - lenkernel + length]

    stimset["gammatones"] = gammatones
    stimset["gammatones_params"] = gparams
    stimset["rubberdrops"] = rubberdrops
    stimset["rubberdrops_params"] = rparams
    stimset["samplerate"] = conf["audio_in"]["rate"]

    # save
    fn = os.path.join(pth, "stimset.h5")
    save_to_h5(fn, gammatones)
    save_to_h5(fn, rubberdrops)
    save_to_h5(fn, stimset)

    # load
    stimset = load_from_h5(fn)
    plot_stimset(plotpth, stimset)
    print(stimset.keys())
    print(stimset["gammatones"].keys())
    print(stimset["gammatones_params"])
    print(stimset["rubberdrops"].keys())
    print(stimset["rubberdrops_params"])

    return


def make_stimulus_set_2():
    """
    Scales rubberdrop sounds to target amplitude
    """

    sr = conf["audio_in"]["rate"]
    obs = {"pressureCenter": get_pressure,
           "acceleration_y": get_acceleration_y,
           "acceleration_x": get_acceleration_x}

    stimset = {}
    repeat = 10

    # record impulse responses for each hydrophone

    print("record impulse responses.")
    rec_irs(pth, repeat=repeat)
    plot_irs(irpth=pth, plot_each=False, obs=obs)

    # make kernels for observables
    print("make kernels.")
    kernels = make_kernels(irpth=pth, obs=obs)
    lenkernel = np.array(kernels["pressureCenter"]).shape[1]  # targets are shifted by length of kernel, don't crop them
    plot_kernels(kpth=pth)

    # crop stimuli
    croptime = int(sr * 0.018)
    print("make rubber drop sounds")
    # Make rubberdrop, correc amps
    rubberdrops, rparams = make_scaled_rubberdrop_sounds(repeat)
    for key, val in rubberdrops.items():
        rubberdrops[key]["sound"] = val["sound"][:, croptime:-croptime]
        if "target" in rubberdrops[key].keys():
            for k, v in rubberdrops[key]["target"].items():
                rubberdrops[key]["target"][k] = v[croptime - lenkernel:-croptime]
    length = rubberdrops[key]["sound"].shape[1]

    stimset["rubberdrops"] = rubberdrops
    stimset["rubberdrops_params"] = rparams
    stimset["samplerate"] = conf["audio_in"]["rate"]

    # save
    fn = os.path.join(pth, "stimset.h5")
    save_to_h5(fn, rubberdrops)
    save_to_h5(fn, stimset)

    # load
    stimset = load_from_h5(fn)
    plot_stimset2(plotpth, stimset)
    print(stimset.keys())
    print(stimset["rubberdrops"].keys())
    print(stimset["rubberdrops_params"])

    return


def plot_stimset(plotpth, stimset):
    sr = conf["audio_in"]["rate"]

    n = len(stimset["gammatones"])
    fig = plt.figure(figsize=(24 / 2.54, 24 / 2.54))
    gs = gridspec.GridSpec(n // 2, 2)
    # sound
    sp0c = 0
    sp1c = 0
    for i, (key, val) in enumerate(sorted(stimset["gammatones"].items())):
        sd = val["sound"]
        if "sp0" in key:
            ax = fig.add_subplot(gs[sp0c, 0])
            sp0c += 1
        if "sp1" in key:
            ax = fig.add_subplot(gs[sp1c, 1])
            sp1c += 1
        t = np.arange(len(sd.T)) / sr
        for i in range(len(sd)):
            ax.plot(t * 1000, sd[i], lw=.7, label=f'speaker {i}')
        ax.set_xlabel("Time [$ms$]")
        ax.set_ylabel("Daq AO (V)")
        ax.set_title(key, fontsize=6)
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plotpth, "gammatones.png"), dpi=300)
    plt.show()

    n = len(stimset["rubberdrops"])
    fig = plt.figure(figsize=(24 / 2.54, 24 / 2.54))
    gs = gridspec.GridSpec(n // 2, 2)
    # sound
    sp0c = 0
    sp1c = 0
    for i, (key, val) in enumerate(sorted(stimset["rubberdrops"].items())):
        sd = val["sound"]
        if "sp0" in key:
            ax = fig.add_subplot(gs[sp0c, 0])
            sp0c += 1
        if "sp1" in key:
            ax = fig.add_subplot(gs[sp1c, 1])
            sp1c += 1
        t = np.arange(len(sd.T)) / sr
        for i in range(len(sd)):
            ax.plot(t * 1000, sd[i], lw=.7, label=f'speaker {i}')
        ax.set_xlabel("Time [$ms$]")
        ax.set_ylabel("Daq AO (V)")
        ax.set_title(key, fontsize=6)
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plotpth, "rubberdrops.png"), dpi=300)
    plt.show()


def plot_stimset2(plotpth, stimset):
    sr = conf["audio_in"]["rate"]
    n = len(stimset["rubberdrops"])
    fig = plt.figure(figsize=(24 / 2.54, 24 / 2.54))
    gs = gridspec.GridSpec(n // 2, 2)
    # sound
    sp0c = 0
    sp1c = 0
    for i, (key, val) in enumerate(sorted(stimset["rubberdrops"].items())):
        sd = val["sound"]
        if "sp0" in key:
            ax = fig.add_subplot(gs[sp0c, 0])
            sp0c += 1
        if "sp1" in key:
            ax = fig.add_subplot(gs[sp1c, 1])
            sp1c += 1
        t = np.arange(len(sd.T)) / sr
        for i in range(len(sd)):
            ax.plot(t * 1000, sd[i], lw=.7, label=f'speaker {i}')
        ax.set_xlabel("Time [$ms$]")
        ax.set_ylabel("Daq AO (V)")
        ax.set_title(key, fontsize=6)
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plotpth, "rubberdrops.png"), dpi=300)
    plt.show()


def make_all_rubberdrop_sounds(pth, repeat):
    # target amplitude for rubber drop (sp0, sp1 : non-inverted, inverted)
    rubberdrops = {}
    rparams = {}
    d, params = make_scaled_rubberdrop_sounds(repeat)
    rubberdrops.update(d)
    rparams.update(params)
    # inverted pressure rubberdrop sounds for all above
    ds = {}
    for key, sound in rubberdrops.items():
        handle = f"{key}_pressureInv"
        d = make_pressure_inversion(pth, handle, sound["sound"], repeat)
        ds.update(d)
    rubberdrops.update(ds)
    return rubberdrops, rparams


def make_scaled_rubberdrop_sounds(repeat):
    d = {}
    # define measurement as target
    obs = {"pressureCenter": get_pressure,
           "acceleration_y": get_acceleration_y,
           "acceleration_x": get_acceleration_x}

    fn = 'rubberdrop_HP100LP20000_1stDrop_1stpart_normalized_smalldelay_shortenedForDAQ_upsampled51200.wav'
    wav, sr = sf.read(fn)
    assert sr == conf["audio_in"]["rate"]
    wav /= np.max(abs(wav))
    wav = np.pad(wav, int(.02 * sr))

    single_speakers = [[1, 0, 0, 0], [0, 1, 0, 0]]
    rparams = {"targetPascal": 562.341325}  # 316.227766
    handles = [fn + f"scaleTo{rparams['targetPascal']:.2f}Pa_sp{np.argwhere(s).squeeze()}" for s in single_speakers]
    for handle, single_speaker in zip(handles, single_speakers):
        # record sound
        rec_repeated(pth, handle, wav, sp=single_speaker, repeat=repeat)
        plot_repeated(pth, handle, obs, plot_each=False)

        meas = measSI_from_rec(pth, handle, obs, lowcut=200, highcut=12000)
        meas_p = np.max(abs(meas["pressureCenter"]))
        print("measured:", meas_p)
        scale_factor = rparams["targetPascal"] / meas_p
        print("scaled:", scale_factor)
        wavscaled = scale_factor * wav
        sound = np.array(single_speaker)[:, None] * wavscaled[None, :]
        d[handle] = {"sound": sound}
        d[handle + "_inv"] = {"sound": -sound}
    return d, rparams


def make_pressure_inversion(pth, handle, basesound, repeat):
    sr = conf["audio_in"]["rate"]
    obs = {"pressureCenter": get_pressure,
           "acceleration_x": get_acceleration_x,
           "acceleration_y": get_acceleration_y}

    if "sp1" in handle:
        single_speaker = [0, 1, 0, 0]
    elif "sp0" in handle:
        single_speaker = [1, 0, 0, 0]

    # record sound
    spidx = np.argwhere(single_speaker).squeeze()

    rec_sound(pth, handle, basesound, repeat=repeat)
    plot_rec_sound(pth, handle, obs)

    # define measurement as target
    # obs = {"pressureCenter": get_pressure,
    #       "acceleration_y": get_acceleration_y}
    meas = measSI_from_rec_sound(pth, handle, obs, lowcut=200, highcut=12000)
    tg = meas
    invert_pressure = 1
    if invert_pressure:
        tg["pressureCenter"] = -1 * tg["pressureCenter"]
        tg["acceleration_y"] = 0 * tg["pressureCenter"]

    # find sound
    sp = [single_speaker, [0, 0, 1, 0], [0, 0, 0, 1]]  # signal to speaker mapping
    kernels = load_kernels(kpth=pth)
    # kernels.pop('acceleration_x')
    sound = find_sound(ks=kernels, tg=tg, sp=sp)
    sound = butter_bandpass_filter(sound, lowcut=200, highcut=12000, fs=sr, order=4, repeat=4)
    delta_offset = int(sr * 0.005)  # remove offset introduced by kernel
    sound = sound[:, delta_offset:]
    tg = {key: val[delta_offset:] for (key, val) in tg.items()}
    save_sound(outpth=pth, handle=handle, sd=sound, ks=kernels, tg=tg)
    rec_sound(outpth=pth, handle=handle, sound=sound, repeat=repeat)
    plot_sound_simulation(pth=pth, handle=handle, recorded=True, obs=obs)
    """
    # add/overwrite with ground truth single speaker
    sound = load_sound(pth=pth, handle=handle)
    obs = {"pressureCenter": get_pressure,
           "acceleration_x": get_acceleration_x,
           "acceleration_y": get_acceleration_y}
    # save again with single speaker added
    #sound["target"] = measSI_from_rec_sound(pth, handle, obs, lowcut=200, highcut=12000)
    #sound["target"]["pressureCenter"] = - sound["target"]["pressureCenter"]
    #sound["target"]["acceleration_x"] = sound["target"]["acceleration_x"]
    #sound["target"]["acceleration_y"] = sound["target"]["acceleration_y"]
    wav = basesound[spidx]
    wav = np.pad(wav, (int(sr * 0.005), 0))
    sound["sound"][spidx] = np.pad(wav, int(.02 * sr))
    save_sound(outpth=pth, handle=handle + "overwrite", sd=sound["sound"], ks=load_kernels(kpth=pth),
               tg=sound["target"])
    # test sound
    rec_sound(outpth=pth, handle=handle + "overwrite", sound=sound["sound"], repeat=repeat)
    plot_sound_simulation(pth=pth, handle=handle + "overwrite", recorded=True, obs=obs)
    """
    d = {}
    d[handle] = load_sound(pth=pth, handle=handle)

    return d


def make_all_gammatone_sounds(pth, repeat):
    sr = conf["audio_in"]["rate"]
    # observables
    obs = {"pressureCenter": get_pressure,
           "acceleration_x": get_acceleration_x,
           "acceleration_y": get_acceleration_y}

    # gammatone
    gparams = {"targetPascal": 316.227766}
    d = {}  # sounds

    # find amplitude gammatone parameter and keep it fixed for different phases
    a = 1
    b = 120
    f = 800
    phi = 0
    duration = 0.013
    padduration = 0.02
    wav, params = gammatone(a=a, b=b, f=f, phi=phi, n=2, duration=duration, padduration=padduration, sr=sr)
    a = gparams["targetPascal"] / np.max(abs(wav))  # rescale a for target max. amplitude
    single_speakers = [[1, 0, 0, 0], [0, 1, 0, 0]]
    handles = [f"gammatone_{gparams['targetPascal']:.2f}Pa_sp{np.argwhere(s).squeeze()}" for s in single_speakers]
    phi_list = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    for basehandle, single_speaker in zip(handles, single_speakers):
        for phi in phi_list:
            wav, params = gammatone(a=a, b=b, f=f, phi=phi, n=2, duration=duration, padduration=padduration, sr=sr)
            r = 0.022  # 0.0175

            if single_speaker == [1, 0, 0, 0]:
                sign = 1
            elif single_speaker == [0, 1, 0, 0]:
                sign = -1
            tg = {"pressureCenter": wav, "acceleration_x": sign * a_sphere(wav, r=r), "acceleration_y": 0 * wav}

            gparams["MonopolAtDistance"] = r
            handle = f"monopole{r * 100:.2f}cm_" + f"phi{phi:.2f}_" + basehandle

            # use three speakers (left or right + opposing orthogonal pair)
            sp = [single_speaker, [0, 0, 1, 0], [0, 0, 0, 1]]
            kernels = load_kernels(kpth=pth)

            sound = find_sound(ks=kernels, tg=tg, sp=sp)
            sound = butter_bandpass_filter(sound, lowcut=200, highcut=12000, fs=sr, order=4, repeat=4)
            save_sound(outpth=pth, handle=handle, sd=sound, ks=kernels, tg=tg)

            # playback signals
            sound = load_sound(pth=pth, handle=handle)
            rec_sound(outpth=pth, handle=handle, sound=sound["sound"], repeat=repeat)
            # plot_rec_sound(pth=pth, handle=handle, obs=obs)
            plot_sound_simulation(pth=pth, handle=handle, recorded=True, obs=obs)
            d[handle] = sound
        params["phi"] = phi_list
    gparams.update(params)
    return d, gparams


def gammatone(a, b, f, phi, n, duration, padduration, sr):
    t = np.arange(int(duration * sr)) / sr
    gammat = a * t ** (n - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * f * t + phi)
    gammat = np.pad(gammat, int(padduration * sr))
    return gammat, locals()


def run():
    # observables
    obs = {"pressure": get_pressure,
           "acceleration_x": get_acceleration_x}

    # record impulse responses for each hydrophone
    rec_irs(pth, repeat=1)
    plot_irs(irpth=pth, plot_each=False, obs=obs)

    # make kernels for observables
    make_kernels(irpth=pth, obs=obs)
    plot_kernels(kpth=pth)

    # define target(s)
    wav = gaussian_double_pulse(samplerate=conf["audio_in"]["rate"], center_frequency=1000, pulse_duration=0.02)
    target_list = [{"pressure": 100 * wav, "acceleration_x": 0 * wav},
                   {"pressure": 0 * wav, "acceleration_x": 1 * wav},
                   {"pressure": 100 * wav, "acceleration_x": a_plane(100 * wav)},
                   {"pressure": 100 * wav, "acceleration_x": a_sphere(100 * wav, r=0.01)}]
    target_handles = ["PressureOnly", "MotionOnly", "PlaneWave", "MonopoleWave1cm"]

    # get signals that produce targets
    sp = get_first_speakers(n=len(obs))  # signal to speaker mapping
    # sp = get_speakers_custom()
    kernels = load_kernels(kpth=pth)

    for handle, tg in zip(target_handles, target_list):
        sound = find_sound(ks=kernels, tg=tg, sp=sp)
        save_sound(outpth=pth, handle=handle, sd=sound, ks=kernels, tg=tg)
        # plot_sound_simulation(pth=pth, handle=handle)

    # playback signals
    for handle, _ in zip(target_handles, target_list):
        sound = load_sound(pth=pth, handle=handle)["sound"]
        rec_sound(outpth=pth, handle=handle, sound=sound, repeat=1)
        # plot_rec_sound(pth=pth, handle=handle, obs=obs)
        plot_sound_simulation(pth=pth, handle=handle, recorded=True, obs=obs)
    return


def save_to_wav(pth, handle):
    import soundfile as sf
    sound = load_sound(pth, handle)
    wav = sound["sound"].T
    sr = sound["daqconfig"]["audio_in"]["rate"]
    sf.write(os.path.join(pth, handle + '.wav'), wav, sr)


def rec_irs(pth, repeat=1):
    """
    Record the impulse response for each individual speaker.
    :param pth: folder path to store recording
    :param repeat: int, repeat pulse playback this often
    :return: dict, recording
    """
    sr = conf["audio_in"]["rate"]
    # for each speaker
    sps = get_first_speakers()
    # kronecker delta pulse
    pp = {"duration": 0.01, "offset": 0.005}
    dp = deltapulse(samplerate=sr, **pp)
    dp = dp / np.max(abs(dp))
    # playback
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=dp.shape[0])
    recs = {}
    # loop over speakers
    for i, sp in enumerate(sps):
        sound = np.array(sp)[:, None] * dp[None, :]
        data_list = []
        # repeats
        for _ in range(repeat):
            data_list.append(daq.play_and_record(sound))
            time.sleep(.1)
        recs[str(i)] = dict(samplerate=sr,
                            recorded=data_list,
                            playback_sound=sound,
                            daqconf=daq.conf,
                            pulse_param=pp)
    # save and close
    fn = os.path.join(pth, 'impulse_responses.h5')
    save_to_h5(fn, recs, verbose=2)
    daq.close()
    return recs


def plot_irs(irpth, plot_each=True, obs=None):
    """
    Plots the impulse responses that were measured with function "rec_irs()"
    :param irpth: folder path to recording of imulse responses
    :param plot_each: whether to plot each repeated pulse playback, or just the mean
    :param obs: dict, definition of observables to turn recording into observables in SI units
    :return:
    """
    fn = os.path.join(irpth, 'impulse_responses.h5')
    recs = load_from_h5(fn)
    for i, rec in enumerate(recs):
        ir_list = []
        for j, data in enumerate(rec["recorded"]):
            ir_list.append(data)
            if plot_each:
                plot_timeseries(f'impulse_responses_rec_{i}_{j}.png', data, rec["samplerate"])
                if obs is not None:
                    plot_timeseries_obs(f'impulse_responses_rec_{i}_{j}_obs.png', data, rec["samplerate"],
                                        rec["playback_sound"], obs)
        ir_mean = np.mean(np.array(ir_list), axis=0)
        mean_over = len(ir_list)
        plot_timeseries(f'impulse_responses_rec_{i}_meanOver{mean_over}.png', ir_mean, rec["samplerate"])
        if obs is not None:
            plot_timeseries_obs(f'impulse_responses_rec_{i}_meanOver{mean_over}_obs.png', ir_mean, rec["samplerate"],
                                rec["playback_sound"], obs)


def make_kernels(irpth, obs):
    """
    Reads recording of pulse playbacks and turns repeated playbacks into a mean response.
    Removes offset of the delta-pulse position.
    :param irpth: folder path to store kernels
    :param obs: dictionary that holds functions that can turn the recording into observables. E.g.
            obs =   {"pressure": get_pressure,
                    "acceleration_x": get_acceleration_x}
    :return: dict, kernels
    """
    fn = os.path.join(irpth, 'impulse_responses.h5')
    recs = load_from_h5(fn)
    # compute mean impulse responses for each speaker
    ir_means = []
    for i, rec in enumerate(recs):
        offset = int(rec["pulse_param"]["offset"] * rec["samplerate"])
        ir_list = []
        for j, data in enumerate(rec["recorded"]):
            # remove DC offsets of recording
            data_filt = butter_highpass_filter(data.T, cut=20, fs=rec["samplerate"], order=4,
                                               repeat=3)  # todo check
            ir_list.append(data_filt.T)
        ir_mean = np.mean(np.array(ir_list), axis=0)
        ir_mean = ir_mean[offset:]
        ir_means.append(ir_mean)

    # turn recording into observables
    kernels = {}
    for i, (key, func) in enumerate(sorted(obs.items())):
        ir_obs = [func(ir) for ir in ir_means]
        kernels[key] = ir_obs
    # save
    fn = os.path.join(irpth, 'kernels.h5')
    save_to_h5(fn, kernels, verbose=2)
    return kernels


def load_kernels(kpth):
    """
    Loads kernels.
    :param kpth: folder path
    :return:
    """
    fn = os.path.join(kpth, 'kernels.h5')
    kernels = load_from_h5(fn)
    return kernels


def plot_kernels(kpth):
    """
    Plots the generated kernels (they are in SI units), after running make_kernels().
    :param kpth: folder path to kernels
    :return:
    """
    sr = conf["audio_in"]["rate"]  # sr is not saved along kernels, check impulse_responses.h5
    kernels = load_kernels(kpth)
    # all observables
    for key, vals in sorted(kernels.items()):
        # all speakers
        for i, val in enumerate(vals):
            plot_timeseries(f'kernels_{key}_speaker{i}.png', val, sr)


def find_sound(ks, tg, sp):
    """
    Computes sound that can generate the desired target signals, given kernels.
    Kernels are given for each speaker. However, one could couple several speakers, a mapping which
    is defined by the sp variable.

    To solve the system of equations in the Fourier domain,
    target signals and kernels are zero-padded to equal length.

    L: Length of target waveform (duration)
    K: Length of impulse response (duration)
    Mt: Number of target waveforms
    Mi: Number of kernels
    M=Mi=Mt, number of target waveforms must be equal to number of kernels

    :param ks: dict, kernels. (key: observable, val: list of kernel waveforms, one for each speaker)
    :param tg: dict, targets. (key: observable, val: target waveforms)
    :param sp: array, speaker choice: picks Mi (combinations of) kernels,
                e.g. [[1,0],[0,1]] for a signal=speaker mapping in the case of two observables and two speakers.
    :return: array, dim: (K+L) x M, sound that generates target waveforms
    """
    # check whether kernels and targets refer to the same observables
    if tg.keys() != ks.keys():
        raise ValueError("Target waveforms are defined in terms of observables for which no kernels are known.")
    # dict to array
    keys = sorted(tg.keys())
    print(keys)
    tgs = np.array([tg[key] for key in keys]).T  # L x Mt
    irs = np.array([np.dot(sp, ks[key]) for key in keys]).T  # K x Mi x Mt
    # zero padding for equal duration
    tgs_pad = np.zeros((len(tgs) + len(irs), *tgs.shape[1:]))  # (K+L) x Mt
    tgs_pad[len(irs):] = tgs  # zeros at beginning
    irs_pad = np.zeros((len(tgs) + len(irs), *irs.shape[1:]))  # (K+L) x Mi x Mt
    irs_pad[:len(irs)] = irs  # zeros at end
    # fourier domain
    tgs_pad_f = np.fft.fft(tgs_pad.T).T  # (K+L) x Mt
    irs_pad_f = np.fft.fft(irs_pad.T).T  # (K+L) x Mi x Mt
    # solve
    if irs_pad_f.ndim == 3:
        irs_pad_f = np.swapaxes(irs_pad_f, 1, 2)
    sg_pad_f = np.linalg.solve(irs_pad_f, tgs_pad_f)  # (K+L) x Mi
    # transform back
    sg_pad = np.real(np.fft.ifft(sg_pad_f.T).T)  # (K+L) x Mi
    sound = np.dot(sg_pad, sp).T  # Number of connected speakers x (K+L)
    return sound


def save_sound(outpth, handle, sd, ks, tg, info=None):
    """
    Save sound, including the kernels and targets that were used to arrive at this sound.
    :param outpth: folder path
    :param handle: str, handle for output file
    :param sd: sound, (number of speakers x duration)
    :param ks: dict, kernels
    :param tg: dict, targets
    :return:
    """
    fn = os.path.join(outpth, f'sound_{handle}.h5')
    sound = {}
    sound["sound"] = sd
    sound["kernel"] = ks
    sound["target"] = tg
    # also save some global configs
    sound["daqconfig"] = conf
    sound["params"] = params
    if info is not None:
        sound["info"] = info
    save_to_h5(fn, sound, verbose=2)
    return


def load_sound(pth, handle):
    """
    Load sound dictionary
    :param pth: folder path
    :param handle: str, handle for output file
    :return:
    """
    fn = os.path.join(pth, f'sound_{handle}.h5')
    sound = load_from_h5(fn)
    return sound


def play_sound(sound):
    """
    Plays sound forever
    :param sound: sound to be played (number of speakers x duration)
    :return:
    """
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=sound.shape[1])
    while True:
        _ = daq.play_and_record(sound)
        time.sleep(.2)
    daq.close()
    return


def rec_sound(outpth, handle, sound, repeat=1, info=None):
    """
    Plays sound and records from all hydrophones.
    :param outpth: str, folder to save results
    :param handle: str, handle for output file
    :param sound: sound to be played (number of speakers x duration)
    :param repeat: repeat playback this often
    :return: dict, recording
    """
    sr = conf["audio_in"]["rate"]
    # initialize DAQ card
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=sound.shape[1])
    # store playbacks
    recordings = dict()
    for i in range(repeat):
        rec = daq.play_and_record(sound)
        d = dict(samplerate=sr,
                 recorded=rec,
                 playback_sound=sound,
                 daqconf=daq.conf,
                 params=params)
        if info is not None:
            d["info"] = info
        recordings[str(i)] = d
        time.sleep(.2)

    # save and close
    fn = os.path.join(outpth, f'sound_{handle}_rec.h5')
    save_to_h5(fn, recordings, verbose=2)
    daq.close()
    return recordings


def plot_rec_sound(pth, handle, obs=None, plot_each=False):
    """
    Plot results of rec_sound()
    :param pth: folder path sound*_rec.h5
    :param handle: str, handle for input files
    :param obs: dict, definition of observables to turn recording into observables in SI units
    :return:
    """
    fn = os.path.join(pth, f'sound_{handle}_rec.h5')
    recs = load_from_h5(fn)
    data_list = []
    for i, rec in enumerate(recs):
        if plot_each:
            plot_timeseries(f'sound_{handle}_rec_{i}.png', rec["recorded"], rec["samplerate"])
            if obs is not None:
                plot_timeseries_obs(f'sound_{handle}_rec_obs_{i}.png', rec["recorded"], rec["samplerate"],
                                    rec["playback_sound"], obs)
        data_list.append(rec["recorded"])
    meanOver = len(data_list)
    mean = np.array(data_list).mean(0)
    plot_timeseries(f'sound_{handle}_rec_meanOver{meanOver}.png', mean, rec["samplerate"])
    if obs is not None:
        plot_timeseries_obs(f'sound_{handle}_rec_obs_meanOver{meanOver}.png', mean, rec["samplerate"],
                            rec["playback_sound"], obs)
    return mean


def plot_sound_simulation(pth, handle, recorded=False, obs=None, sound=None, ks=None):
    """
    Plots the speaker activations that can produce targets.
    Collects results from the sound dictionary, following find_sound and save_sound().
    If one has already played back the conditioned sound via rec_sound(), and one sets recorded=True,
    the results of the playback are plotted.
    A convolution of the speaker activation with the kernel is run,
    to predict whether the target waveform should be generated by the signal.

    :param pth: folder path to sound*.h5 (and sound*_rec.h5)
    :param handle: str, handle for input files
    :param recorded: whether to look for recordings
    :param obs: dict, definition of observables to turn recording into observables in SI units
    :return:
    """
    sr = conf["audio_in"]["rate"]
    if sound is None:
        sound = load_sound(pth, handle)
    sd = sound["sound"]  # number of speakers x (K+L)
    if ks is None:
        ks = sound["kernel"]  # dict, number of speakers x K
    tg = sound["target"]  # dict, L

    if recorded:
        fn = os.path.join(pth, f'sound_{handle}_rec.h5')
        recs = load_from_h5(fn)

    nobs = len(tg)
    fig = plt.figure(figsize=(24 / 2.54, (6 * nobs + 1) / 2.54))
    gs = gridspec.GridSpec(nobs + 1, 1)
    # sound
    ax0 = fig.add_subplot(gs[0, 0])
    t = 1000. * np.arange(sd.shape[1], dtype=float) / sr  # in ms
    for i, s in enumerate(sd):
        ax0.plot(t, s, lw=1, alpha=.6, label=f'signal (a.u.), speaker {i}')
    ax0.spines['right'].set_visible(False), ax0.spines['top'].set_visible(False)
    ax0.set_xlabel(r'Time [$ms$]')
    ax0.set_ylabel(r'Voltage to Amplifier')
    ax0.legend(loc='upper right', fontsize=10)
    # targets
    for i, (key, val) in enumerate(sorted(tg.items())):
        # prediction: convolution of sound with kernel for each speaker
        kernel = np.array(ks[key])
        prediction = np.sum(np.array([np.convolve(sd[j], kernel[j]) for j in range(sd.shape[0])]), axis=0)

        shift = kernel.shape[1]
        ax = fig.add_subplot(gs[i + 1, 0])
        t = 1000. * np.arange(shift, val.shape[0] + shift, dtype=float) / sr  # in ms
        ax.plot(t, val, label="target", lw=1)

        t = 1000. * np.arange(prediction.shape[0], dtype=float) / sr  # in ms
        ax.plot(t, prediction, label="prediction", alpha=.6, lw=.8)

        if recorded and obs is not None:
            data_list = []
            for i, rec in enumerate(recs):
                data_list.append(rec["recorded"])
            data = np.array(data_list).mean(axis=0)
            t = 1000. * np.arange(data.shape[0], dtype=float) / sr  # in ms
            func = obs[key]
            ax.plot(t, func(data), lw=.7, alpha=0.4, c='k', label="measured")

        ax.set_xlabel(r'Time [$ms$]')
        ax.set_ylabel(key)
        ax.legend(loc='upper right', fontsize=10)
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax0.set_xlim(ax.get_xlim())
    # save
    fn = f'soundPred_{handle}.png'
    fig.savefig(os.path.join(plotpth, fn), dpi=300)
    plt.close('all')


def rec_repeated(pth, handle, base_wav, sp=None, repeat=1, info=None, save=True):
    sr = conf["audio_in"]["rate"]
    # initialize DAQ card
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=base_wav.shape[0])
    # store playbacks
    recordings = dict()
    # playback routine
    if sp is None:
        sp = get_first_speakers()[0]

    for i in range(repeat):
        print(sp)
        # play and record
        sound = np.array(sp)[:, None] * base_wav[None, :]
        rec = daq.play_and_record(sound)
        # save
        d = dict(samplerate=sr,
                 recorded=rec,
                 playback_sound=sound,
                 daqconf=daq.conf,
                 params=params, )
        if info is not None:
            d["info"] = info
        recordings[str(i)] = d
        time.sleep(.3)

    # save and close
    if save:
        fn = os.path.join(pth, f'rec_{handle}.h5')
        save_to_h5(fn, recordings, verbose=2)
    daq.close()
    return [value for _, value in recordings.items()]


def measSI_from_rec(pth, handle, obs, lowcut=None, highcut=None):
    sr = conf["audio_in"]["rate"]
    fn = os.path.join(pth, f'rec_{handle}.h5')
    recs = load_from_h5(fn)
    data_list = []
    for i, rec in enumerate(recs):
        data = rec["recorded"]
        if lowcut is not None and highcut is not None:
            data = butter_bandpass_filter(data.T, lowcut=lowcut, highcut=highcut, fs=sr, order=4, repeat=4).T
        data_list.append(data)
    mean = np.array(data_list).mean(0)
    meas = {}
    for i, (key, func) in enumerate(sorted(obs.items())):
        meas[key] = func(mean)
    return meas


def measSI_from_rec_sound(pth, handle, obs, lowcut=None, highcut=None):
    sr = conf["audio_in"]["rate"]
    fn = os.path.join(pth, f'sound_{handle}_rec.h5')
    recs = load_from_h5(fn)
    data_list = []
    for i, rec in enumerate(recs):
        data = rec["recorded"]
        if lowcut is not None and highcut is not None:
            data = butter_bandpass_filter(data.T, lowcut=lowcut, highcut=highcut, fs=sr, order=4, repeat=4).T
        data_list.append(data)
    mean = np.array(data_list).mean(0)
    meas = {}
    for i, (key, func) in enumerate(sorted(obs.items())):
        meas[key] = func(mean)
    return meas


def plot_repeated(pth, handle, obs, plot_each=True, recs=None, show=False, show_fft_coefs=False, fft_cutoff=5000):
    if recs is None:
        fn = os.path.join(pth, f'rec_{handle}.h5')
        recs = load_from_h5(fn)
    data_list = []
    for i, rec in enumerate(recs):
        if plot_each:
            plot_timeseries(f'rec_{handle}_{i}.png', rec["recorded"], rec["samplerate"], show_fft_coefs=show_fft_coefs, fft_cutoff=fft_cutoff)
            if obs is not None:
                plot_timeseries_obs(f'rec_{handle}_obs_{i}.png', rec["recorded"], rec["samplerate"],
                                    rec["playback_sound"], obs,show_fft_coefs=show_fft_coefs,fft_cutoff=fft_cutoff)
        data_list.append(rec["recorded"])
    data_mean = np.array(data_list).mean(0)
    mean_over = len(data_list)
    plot_timeseries(f'rec_{handle}_meanOver{mean_over}.png', data_mean, rec["samplerate"],show_fft_coefs=show_fft_coefs,fft_cutoff=fft_cutoff)

    if obs is not None:
        plot_timeseries_obs(f'rec_{handle}_obs_meanOver{mean_over}.png', data_mean,
                            rec["samplerate"],
                            rec["playback_sound"], obs=obs, show=show,show_fft_coefs=show_fft_coefs,fft_cutoff=fft_cutoff)
    return


def a_plane(p):
    sr = conf["audio_in"]["rate"]
    # acceleration from pressure for a plane wave
    omega = 2 * np.pi * np.fft.fftfreq(p.shape[-1]) * sr
    derivative = -1j * omega
    a = np.real(np.fft.ifft(np.fft.fft(p) * derivative / (params["rho"] * params["c"])))
    return a


def p_plane(wav):
    # todo
    return wav


def a_sphere(p, r=0.1):
    """
    see e.g. Eq A3 in https://doi.org/10.1242/jeb.093831, v = (1+i/kr)*p/(rho*c)
    :param p:
    :param r:
    :return:
    """
    sr = conf["audio_in"]["rate"]
    # radial acceleration from pressure for a spherical wave
    k = 2 * np.pi * np.fft.fftfreq(p.shape[-1])[1:] * sr / params[
        "c"]  # exclude DC component (divergence of 1/kr)
    coeff = 1 + 1j / (k * r)
    derivative = -2 * np.pi * 1j * np.fft.fftfreq(p.shape[-1])[1:] * sr
    tmp = np.fft.fft(p)[1:] * coeff * derivative
    a = np.real(np.fft.ifft(np.insert(tmp, 0, 0) / (params["rho"] * params["c"])))  # insert zero DC
    return a


def p_sphere(wav, r=0.1):
    # todo
    return wav


def gaussian_double_pulse(samplerate, center_frequency=5000, pulse_duration=0.001):
    """
    Creates a gausian double pulse with a given center frequency
    """
    tt = np.arange(0, int(pulse_duration * samplerate), dtype=float)
    wav = np.sin(1 * np.pi * 1e3 * tt / samplerate)
    ts = 1 / (2 * np.pi * center_frequency) * samplerate
    ct = (pulse_duration * samplerate) // 2  # 10 * ts
    wav = (tt - ct) / ts * np.exp(-((tt - ct) / ts) ** 2 / 2)
    pulse = wav / wav.max()
    return pulse


def pulsed_sinusoid(samplerate, freq=200, duration=0.001):
    """
    Creates a sinusoid pulse in a hann window
    """
    tt = np.arange(0, int(duration * samplerate), dtype=float)
    wav = np.sin(2 * np.pi * freq * tt / samplerate)
    wav *= sig.windows.hann(len(wav))
    return wav / wav.max()


def ramped_sinusoid(samplerate, freq=200, duration=0.1, ramp_time=0.01):
    """
    Creates a hann-ramped steady sinusoid
    """
    ramp_samples = int(ramp_time * samplerate)
    tt = np.arange(0, int(duration * samplerate), dtype=float)
    wav = np.sin(2 * np.pi * freq * tt / samplerate)
    hann = sig.windows.hann(2 * ramp_samples + 1)
    wav[:ramp_samples] *= hann[:ramp_samples]
    wav[-ramp_samples:] *= hann[-ramp_samples:]
    return wav / wav.max()


def filtered_deltapulse(samplerate, low=200, high=15000, pulse_duration=0.01):
    """
    Creates a bandpass-filtered delta pulse
    """
    nsp = int(pulse_duration * samplerate)
    tt = np.arange(0, nsp, dtype=float)
    wav = np.zeros(nsp)
    start = int(0.005 * samplerate)
    wav[start] = 1
    wav = butter_bandpass_filter(wav, low, high, samplerate, order=5)
    pulse = wav / wav.max()
    return pulse, start


def deltapulse(samplerate, duration=0.01, offset=0.005):
    """
    Create a kronecker delta pulse
    """
    nsp = int(duration * samplerate)
    start = int(offset * samplerate)
    wav = np.zeros(nsp)
    wav[start] = 1
    return wav


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, repeat=4):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = data
    for _ in range(repeat):
        y = filtfilt(b, a, y)
    return y


def butter_highpass_filter(data, cut, fs, order=4, repeat=4):
    def butter_highpass(cut, fs, order=4):
        nyq = 0.5 * fs
        cut = cut / nyq
        b, a = butter(order, cut, btype='highpass')
        return b, a

    b, a = butter_highpass(cut, fs, order=order)
    y = data
    for _ in range(repeat):
        y = filtfilt(b, a, y)
    return y


def play_simple_function(base_wav=None):
    sr = conf["audio_in"]["rate"]
    # base sound
    if base_wav is None:
        #base_wav = .1 * ramped_sinusoid(sr,800,5,0.1)
        base_wav = .5 * gaussian_double_pulse(samplerate=sr, center_frequency=800, pulse_duration=0.03)
        #base_wav =  deltapulse(samplerate=sr,duration= 0.03, offset=0.005)

    # initialize DAQ card
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=base_wav.shape[0])

    # playback routine
    speaker_choice = [0, 0, 0, 1]  # wav loadings for output channel
    while True:
        sound = np.array(speaker_choice)[:, None] * base_wav[None, :]
        daq.play_and_record(sound)
        time.sleep(.01)
    # close
    # daq.close()
    return


def rec_simple_function(base_wav=None, speaker_choices=None, repeats=1, meta=""):
    # base sound
    sr = conf["audio_in"]["rate"]
    if base_wav is None:
        base_wav = gaussian_double_pulse(samplerate=sr, center_frequency=700, pulse_duration=0.03)
        #base_wav = deltapulse(samplerate=sr, duration=0.03, offset=0.005)
    # initialize DAQ card
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=base_wav.shape[0])
    # store playbacks
    recordings = dict()
    # playback routine
    if speaker_choices is None:
        speaker_choices = get_first_speakers()
    for i, speaker_choice in enumerate(speaker_choices):
        print(speaker_choice)
        # play and record
        sound = np.array(speaker_choice)[:, None] * base_wav[None, :]
        # save
        recs = []
        for j in range(repeats):
            print(j)
            recs.append(daq.play_and_record(sound))
        recordings[str(i)] = dict(samplerate=sr,
                                  recorded=recs,
                                  playback_sound=sound,
                                  daqconf=daq.conf,
                                  params=params,
                                  meta=meta)
        time.sleep(.3)

    # save and close
    fn = os.path.join(pth, 'simple_function_rec.h5')
    save_to_h5(fn, recordings, verbose=2)
    daq.close()
    return recordings

def rec_simple_function_across_amps(base_wav=None, speaker_choices=None, repeats=1, meta="", amplitudes=[1]):
    # base sound
    sr = conf["audio_in"]["rate"]
    if base_wav is None:
        base_wav = gaussian_double_pulse(samplerate=sr, center_frequency=700, pulse_duration=0.06)
        #base_wav = deltapulse(samplerate=sr, duration=0.03, offset=0.005)

    base_wav /= abs(base_wav).max()
    # initialize DAQ card
    daq = DAQPlaybackMultiStim(conf=conf, wavsamples=base_wav.shape[0])
    # store playbacks
    recordings = dict()
    # playback routine
    if speaker_choices is None:
        speaker_choices = get_first_speakers()
    for i, speaker_choice in enumerate(speaker_choices):
        recs = []
        print(speaker_choice)
        for amp in amplitudes:
            print(amp)
            # play and record
            sound = np.array(speaker_choice)[:, None] * amp * base_wav[None, :]
            # save
            reps = []
            for j in range(repeats):
                reps.append(daq.play_and_record(sound))
                print(j)
            recs.append(reps)
        recordings[str(i)] = dict(samplerate=sr,
                                  recorded=recs,
                                  playback_sound=sound,
                                  daqconf=daq.conf,
                                  params=params,
                                  meta=meta,
                                  amplitudes=amplitudes)
        time.sleep(.3)

    # save and close
    fn = os.path.join(pth, 'simple_function_rec_across_amplitudes.h5')
    save_to_h5(fn, recordings, verbose=2)
    daq.close()
    return recordings


def plot_simple_function(obs=None):
    fn = os.path.join(plotpath, 'simple_function_rec.h5')
    recs = load_from_h5(fn)
    for i, rec in enumerate(recs):
        plot_timeseries(f'simple_function_rec_{i}.png', rec["recorded"], rec["samplerate"])
        if obs is not None:
            plot_timeseries_obs(f'simple_function_rec_obs_{i}.png', rec["recorded"], rec["samplerate"],
                                rec["playback_sound"], obs)


def plot_timeseries_obs(fn, data, sr, sound, obs, show=False, show_fft_coefs=False,fft_cutoff=5000):
    """
    Plots observables
    :param fn: str, name for output plot
    :param data: dict, recording
    :param sr: samplerate
    :param sound: played back sound
    :param obs: dict, definition of observables to turn recording into observables in SI units
    :return:
    """
    # time
    nsp = data.shape[0]
    t = 1000. * np.arange(nsp, dtype=float) / sr  # in ms
    # observations
    nobs = len(obs)

    fig = plt.figure(figsize=(24 / 2.54, (6 * nobs + 6) / 2.54), num=123)
    gs = gridspec.GridSpec(nobs + 1, 1)
    # gs.update(left=0.05, right=0.9, top=0.99, bottom=0.3, hspace=.1, wspace=.1)

    # sound
    ax = fig.add_subplot(gs[0, 0])
    for i, s in enumerate(sound):
        if show_fft_coefs:
            f, p = fftcoefs(s, sr)
            hf = f>fft_cutoff
            cutoff = int(np.argwhere(hf)[0])
            ax.plot(f[:cutoff], p[:cutoff], lw=1, alpha=.6, label=f'signal (a.u.), speaker {i}'), ax.spines['right'].set_visible(False), \
                ax.spines['top'].set_visible(False)
        else:
            ax.plot(t, s, lw=1, alpha=.6, label=f'signal (a.u.), speaker {i}'), ax.spines['right'].set_visible(False), \
                ax.spines['top'].set_visible(False)

    ax.set_xlabel(r'Time [$ms$]')
    ax.set_ylabel(r'Voltage to Amplifier')
    ax.legend(loc='upper right', fontsize=10)

    # observables
    for i, (key, func) in enumerate(sorted(obs.items())):
        ax = fig.add_subplot(gs[i + 1, 0])

        if show_fft_coefs:
            f, p = fftcoefs(func(data), sr)
            hf = f>fft_cutoff
            cutoff = int(np.argwhere(hf)[0])
            ax.plot(f[:cutoff], p[:cutoff], lw=1, c='k')
            ax.set_xlabel(r'Frequency [$Hz$]')
        else:
            ax.plot(t, func(data), lw=1, c='k')
            ax.set_xlabel(r'Time [$ms$]')
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
        ax.set_ylabel(key)

    # show or save
    fig.savefig(os.path.join(plotpth, fn), dpi=300)
    if show:
        img = cv2.imread(os.path.join(plotpth, fn))
        img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
        cv2.imshow('image', img)
        cv2.waitKey(1)
    else:
        plt.close('all')
    return

def fftcoefs(x, sr):
    from pylab import rfft, rfftfreq
    dt = 1/sr                                # Define the sampling interval.
    N = x.shape[0]                           # Define the total number of data points.
    #T = N * dt                               # Define the total duration of the data.
    #x  = hanning(N) * x                      # Apply the Hanning taper to the data.
    return rfftfreq(len(x), d = dt), abs(rfft(x))
def plot_timeseries(fn, data, samplerate, labels=None, show_fft_coefs=False, fft_cutoff=5000):
    """
    Waveform subplot for each channel
    :param fn: str, name for output plot
    :param data: dict, recording
    :param samplerate:
    :return:
    """
    nsp = data.shape[0]
    nch = 1
    if data.ndim > 1:
        nch = data.shape[1]
    fig = plt.figure(figsize=(24 / 2.54, 6 * nch / 2.54))
    gs = gridspec.GridSpec(nch, 1)
    # gs.update(left=0.05, right=0.9, top=0.99, bottom=0.3, hspace=.1, wspace=.1)
    # plot
    for i in range(nch):
        if data.ndim > 1:
            ch = data[:, i]
        else:
            ch = data
        ax = fig.add_subplot(gs[i, 0])
        if show_fft_coefs:
            f, p = fftcoefs(ch, samplerate)
            hf = f>fft_cutoff
            cutoff = int(np.argwhere(hf)[0])
            ax.plot(f[:cutoff], p[:cutoff], color='k', lw=1)  # Plot spectrum vs frequency,
            ax.set_xlabel('Frequency [Hz]')

        else:
            ax.plot(1000. * np.arange(nsp, dtype=float) / samplerate, ch, color='k', lw=1)
            ax.set_xlabel('Time [ms]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.savefig(os.path.join(plotpth, fn), dpi=300)
    # plt.show()
    plt.close('all')


class DAQPlaybackMultiStim:
    def __init__(self, conf, wavsamples):
        """
        """
        self.conf = conf
        self.wavsamples = wavsamples
        self.sr = self.conf['audio_in']['rate']
        self.peak_amp = self.conf['audio_in']['peakAmp']
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
            trigger_source="/" + self.conf["audio_in"]["clockStr"] + "/ai/StartTrigger")
        self.writer = AnalogMultiChannelWriter(self.taskAO.out_stream)

    def create_in_task(self):
        self.taskAI = nidaqmx.Task()
        all_chans = sorted(self.conf["audio_in"]["chStr"] + self.conf["accel_in"]["chStr"])

        for ch in all_chans:
            if "audio_in" in self.conf:
                if ch in self.conf["audio_in"]["chStr"]:
                    self.taskAI.ai_channels.add_ai_voltage_chan(ch, max_val=self.peak_amp, min_val=-self.peak_amp)

            if "accel_in" in self.conf:
                if ch in self.conf["accel_in"]["chStr"]:
                    self.taskAI.ai_channels.add_ai_accel_chan(ch, max_val=self.peak_amp,
                                                              min_val=-self.peak_amp,
                                                              sensitivity=self.conf["accel_in"]["sensitivity_lookup"][
                                                                  ch],
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


def save_to_h5(data_fn, data_item, verbose=0):
    """
    saving a nested dictionary data structure to hdf5
    Contents may be: dict, iterables, numpy arrays, str, bytes
    """

    from collections.abc import Iterable

    def iterable(obj):
        return isinstance(obj, Iterable)

    def recursively_save_contents_to_group(h5file, path, data_item):
        """
        The function saves data of dictionaries and lists to hdf5

        Limitation:
        - data labeled with integer-strings will be restored into a list
        """

        def save_key_item(key, value, path):
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float)):
                # if verbose > 0:
                #    print('saving entry: {} -- {}'.format(path + key, type(item)))
                h5file[path + key] = item
            elif isinstance(item, pd.DataFrame):
                item.to_hdf(data_fn, key)
            elif isinstance(item, dict) or iterable(item):
                # print(key, type(key))
                recursively_save_contents_to_group(h5file, path + key + '/', item)

            else:
                raise ValueError('Cannot save %s type' % type(item))

        # dictionaries ...
        if isinstance(data_item, (dict)):
            for key, item in data_item.items():
                save_key_item(key, item, path)

        # iterables ...
        else:
            for j, item in enumerate(data_item):
                key = str(j)
                save_key_item(key, item, path)

    # initial check for data type
    if not (isinstance(data_item, dict) or iterable(data_item)):
        raise ValueError('Cannot save %s type' % type(data_item))

    with h5.File(data_fn, 'w') as h5file:
        recursively_save_contents_to_group(h5file, '/', data_item)


def load_from_h5(filename, verbose=0):
    """
    The function loads a data structure from a hdf5 file to list or dictionary
    """

    def recursively_load_contents_from_group(h5file, path):
        """
        load dictionaries or iterables
        """

        key = list(h5file[path].keys())[0]
        isnumber = False
        try:
            int(key)  # this fails for non-integer strings
            isnumber = True
        except ValueError:
            pass

        # iterables
        if isnumber:
            ans = list()
            keys = [key for key in h5file[path].keys()]
            keys_argsorted = np.argsort([int(key) for key in keys])
            for keyx in keys_argsorted:
                key = keys[keyx]
                item = h5file[path][key]
                if isinstance(item, h5._hl.dataset.Dataset):
                    if verbose > 0:
                        print('loading: {}/{}'.format(path, key))
                    ans.append(item[()])
                elif isinstance(item, h5._hl.group.Group):
                    ans.append(recursively_load_contents_from_group(h5file, path + key + '/'))
        # dictionaries
        else:
            ans = dict()
            for key, item in h5file[path].items():
                if 'pandas_type' in item.attrs.keys():
                    ans[key] = pd.read_hdf(filename, key)
                elif isinstance(item, h5._hl.dataset.Dataset):
                    if verbose > 0:
                        print('loading: {}/{}'.format(path, key))
                    ans[key] = item[()]
                elif isinstance(item, h5._hl.group.Group):
                    ans[key] = recursively_load_contents_from_group(h5file, path + key + '/')
        return ans

    with h5.File(filename, 'r') as h5file:
        return recursively_load_contents_from_group(h5file, '/')


def signal_sweep():
    """
    Example for a signal sweep: Tones and pulses at different frequencies, with and without sound targeting.
    """
    # observables
    obs = {"pressure": get_pressure,
           "acceleration_x": get_acceleration_x}

    # sound set
    dursine = 2
    durpulse = 0.1
    sinef = [15, 30, 60, 100, 120, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000,
             3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    pulsef = [100, 120, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000,
              5000, 6000, 7000, 8000, 9000, 10000]
    sines = [ramped_sinusoid(samplerate=conf["audio_in"]["rate"], freq=f, duration=dursine, ramp_time=0.01) for f in
             sinef]
    pulses = [gaussian_double_pulse(samplerate=conf["audio_in"]["rate"], center_frequency=f, pulse_duration=durpulse)
              for f
              in pulsef]

    # with signal conditioning
    # record impulse responses for each hydrophone
    rec_irs(pth, repeat=10)
    plot_irs(irpth=pth, plot_each=False, obs=obs)
    # make kernels for observables
    make_kernels(irpth=pth, obs=obs)
    plot_kernels(kpth=pth)

    # A) Two Speakers, Two Observables
    # sines
    amp = 100
    for f, wav in zip(sinef[5:], sines[5:]):
        handle = f"sine{f}"
        wav = amp * wav  # in Pa
        target_list = [{"pressure": wav, "acceleration_x": null(wav)},
                       {"pressure": wav, "acceleration_x": a_plane(wav)}]
        target_names = ["pressureCond", "planeCond"]

        # get signals that produce targets
        sp = get_first_speakers(n=len(obs))  # signal to speaker mapping
        kernels = load_kernels(kpth=pth)

        # find sounds
        for name, tg in zip(target_names, target_list):
            handle2 = handle + name
            info = {"stimulus_type": "sinusoid",
                    "stimulus_freq": f,
                    "duration": dursine,
                    "mode": name,
                    "amp": amp}
            sound = find_sound(ks=kernels, tg=tg, sp=sp)
            save_sound(outpth=pth, handle=handle2, sd=sound, ks=kernels, tg=tg, info=info)
            # plot_sound_simulation(pth=pth, handle=handle2)

        # playback signals
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "sinusoid",
                    "stimulus_freq": f,
                    "duration": dursine,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = load_sound(pth=pth, handle=handle2)["sound"]
            rec_sound(outpth=pth, handle=handle2, sound=sound, repeat=1, info=info)
            plot_rec_sound(pth=pth, handle=handle2, obs=obs, plot_each=False)
            plot_sound_simulation(pth=pth, handle=handle2, recorded=True, obs=obs)

    # pulses
    amp = 100
    for f, wav in zip(pulsef[3:], pulses[3:]):
        handle = f"gpulse{f}"
        wav = amp * wav  # in Pa
        target_list = [{"pressure": wav, "acceleration_x": null(wav)},
                       {"pressure": wav, "acceleration_x": a_plane(wav)}]
        target_names = ["pressureCond", "planeCond"]

        # get signals that produce targets
        sp = get_first_speakers(n=len(obs))  # signal to speaker mapping
        kernels = load_kernels(kpth=pth)

        # find sounds
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "gaussian_pulse",
                    "stimulus_freq": f,
                    "duration": durpulse,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = find_sound(ks=kernels, tg=tg, sp=sp)
            save_sound(outpth=pth, handle=handle2, sd=sound, ks=kernels, tg=tg, info=info)
            # plot_sound_simulation(pth=pth, handle=handle2)

        # playback signals
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "gaussian_pulse",
                    "stimulus_freq": f,
                    "duration": durpulse,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = load_sound(pth=pth, handle=handle2)["sound"]
            rec_sound(outpth=pth, handle=handle2, sound=sound, repeat=10, info=info)
            plot_rec_sound(pth=pth, handle=handle2, obs=obs, plot_each=False)
            plot_sound_simulation(pth=pth, handle=handle2, recorded=True, obs=obs)

    # B) Two Speakers, One Observable (extracted from kernels)
    # sines
    amp = 100
    for f, wav in zip(sinef, sines):
        handle = f"sine{f}"
        wav = amp * wav  # in Pa
        target_list = [{"pressure": wav}]
        target_names = ["OnlyPressureControlled"]

        # get signals that produce targets
        sp = [[1, 1]]  # signal to speaker mapping
        kernels = load_kernels(kpth=pth)
        kernels_p = {"pressure": kernels["pressure"]}

        # find sounds
        for name, tg in zip(target_names, target_list):
            handle2 = handle + name
            info = {"stimulus_type": "sinusoid",
                    "stimulus_freq": f,
                    "duration": dursine,
                    "mode": name,
                    "amp": amp}
            sound = find_sound(ks=kernels_p, tg=tg, sp=sp)
            save_sound(outpth=pth, handle=handle2, sd=sound, ks=kernels, tg=tg, info=info)
            # plot_sound_simulation(pth=pth, handle=handle2)

        # playback signals
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "sinusoid",
                    "stimulus_freq": f,
                    "duration": dursine,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = load_sound(pth=pth, handle=handle2)["sound"]
            rec_sound(outpth=pth, handle=handle2, sound=sound, repeat=1, info=info)
            plot_rec_sound(pth=pth, handle=handle2, obs=obs, plot_each=False)
            plot_sound_simulation(pth=pth, handle=handle2, recorded=True, obs=obs)

    # pulses
    amp = 100
    for f, wav in zip(pulsef, pulses):
        handle = f"gpulse{f}"
        wav = amp * wav  # in Pa
        target_list = [{"pressure": wav}]
        target_names = ["OnlyPressureControlled"]

        # get signals that produce targets
        sp = [[1, 1]]  # signal to speaker mapping
        kernels = load_kernels(kpth=pth)
        kernels_p = {"pressure": kernels["pressure"]}

        # find sounds
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "gaussian_pulse",
                    "stimulus_freq": f,
                    "duration": durpulse,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = find_sound(ks=kernels_p, tg=tg, sp=sp)
            save_sound(outpth=pth, handle=handle2, sd=sound, ks=kernels, tg=tg, info=info)
            # plot_sound_simulation(pth=pth, handle=handle2)

        # playback signals
        for name, tg in zip(target_names, target_list):
            info = {"stimulus_type": "gaussian_pulse",
                    "stimulus_freq": f,
                    "duration": durpulse,
                    "mode": name,
                    "amp": amp}
            handle2 = handle + name
            sound = load_sound(pth=pth, handle=handle2)["sound"]
            rec_sound(outpth=pth, handle=handle2, sound=sound, repeat=10, info=info)
            plot_rec_sound(pth=pth, handle=handle2, obs=obs, plot_each=False)
            plot_sound_simulation(pth=pth, handle=handle2, recorded=True, obs=obs)

    # ____
    # without signal conditioning
    # observables
    obs = {"pressure": get_pressure,
           "acceleration_x": get_acceleration_x}
    # sines
    amp = 1
    sp = [1, 0]
    for f, wav in zip(sinef, sines):
        wav = amp * wav  # in V
        handle = f"speaker0_sine{f}"
        info = {"stimulus_type": "sinusoid",
                "stimulus_freq": f,
                "duration": dursine,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=1, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    # pulses
    amp = 1
    sp = [1, 0]
    for f, wav in zip(pulsef, pulses):
        wav = amp * wav  # in V
        handle = f"speaker0_gpulse{f}"
        info = {"stimulus_type": "gaussian_pulse",
                "stimulus_freq": f,
                "duration": durpulse,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=10, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    # sines
    amp = 1
    sp = [0, 1]
    for f, wav in zip(sinef, sines):
        wav = amp * wav  # in V
        handle = f"speaker1_sine{f}"
        info = {"stimulus_type": "sinusoid",
                "stimulus_freq": f,
                "duration": dursine,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=1, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    # pulses
    amp = 1
    sp = [0, 1]
    for f, wav in zip(pulsef, pulses):
        wav = amp * wav  # in V
        handle = f"speaker1_gpulse{f}"
        info = {"stimulus_type": "gaussian_pulse",
                "stimulus_freq": f,
                "duration": durpulse,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=10, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    # sines (paired)
    amp = 1
    sp = [.5, .5]
    for f, wav in zip(sinef, sines):
        wav = amp * wav  # in V
        handle = f"paired_sine{f}"
        info = {"stimulus_type": "sinusoid",
                "stimulus_freq": f,
                "duration": dursine,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=1, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    # pulses (paired)
    amp = 1
    sp = [.5, .5]
    for f, wav in zip(pulsef, pulses):
        wav = amp * wav  # in V
        handle = f"paired_gpulse{f}"
        info = {"stimulus_type": "gaussian_pulse",
                "stimulus_freq": f,
                "duration": durpulse,
                "amp": amp}
        rec_repeated(pth, handle, wav, sp=sp, repeat=10, info=info)
        plot_repeated(pth, handle, obs, plot_each=False)

    return


if __name__ == "__main__":
    main()
