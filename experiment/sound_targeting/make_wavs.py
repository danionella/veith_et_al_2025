import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import glob
import scipy.signal as sig


def main():
    sr = 51200
    outdir = r"./wav_folder"
    gammatone_wavs(outdir, sr)
    plot_all_wavs(outdir)


def plot_all_wavs(dir):
    fns = sorted(glob.glob(os.path.join(dir, "*wav")))
    n = len(fns)
    fig, axs = plt.subplots(n, sharex=True, figsize=(8, 14))
    for i, fn in enumerate(fns):
        name = os.path.split(fn)[1]
        wav, sr = sf.read(fn)
        t = np.arange(len(wav)) / sr
        axs[i].plot(1e3 * t, wav, 'k')
        axs[i].set_title(name, fontsize=8)
        axs[i].set_ylim([-1, 1])
    axs[i].set_xlabel("Time [ms]")
    fig.tight_layout()
    fig.savefig(os.path.join(dir, "plot.png"))
    fig.show()


def gammatone(a, b, f, phi, n, duration, padduration, sr):
    t = np.arange(int(duration * sr)) / sr
    gammat = a * t ** (n - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * f * t + phi)
    gammat = np.pad(gammat, int(padduration * sr))
    return gammat, locals()


def gammatone_wavs(outdir, sr):
    phi_list = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 4 phases
    phi_list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]  # 8 phases
    f_list = [400, 800, 1200, 2000, 5000]
    b = 120
    phi = 0
    n = 2
    duration = 0.012
    padduration = 0.0013

    # choose a that normalizes the sound
    for f in f_list:
        wav, params = gammatone(a=1, b=b, f=f, phi=phi, n=n, duration=duration, padduration=padduration, sr=sr)
        a = 1 / np.max(abs(wav))  # round to 3digits
        for phi in phi_list:
            wav, params = gammatone(a=a, b=b, f=f, phi=phi, n=n, duration=duration, padduration=padduration, sr=sr)
            name = f"gammatone_a{a:.3f}_b{b}_f{f}_n{n}_phi{phi}_dur{duration}_paddur{padduration}.wav"
            name = f"gammatone_{f}Hz_phi{phi / np.pi}pi.wav"  # shortened name
            sf.write(os.path.join(outdir, name), wav, sr)
    return wav, params


if __name__ == "__main__":
    main()
