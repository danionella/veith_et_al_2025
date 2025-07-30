from queue import Empty
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime


class LiveClickCount:
    """
    Live Click Count
    Example config for Danionella click detection:
    "audioFunc": {
    "active": true,
    "className": "LiveClickCount",
    "lowcut": 3000,
    "highcut": 14000,
    "snr_threshold": 10,
    "chStr_use": ["AI0/ai0", "AI0/ai1"],
    "chMixMethod": "max",
    "bin_seconds": 60,
    "allowSaving": true}
    """

    def __init__(self, conf):
        plt.ion()  # Turn on interactive mode
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # hack to open matplotlib window

        # click detection variables
        self.conf = conf
        self.sr = conf['audio']['rate']
        self.lowcut = conf['audioFunc']['lowcut']
        self.highcut = conf['audioFunc']['highcut']
        self.snr_threshold = conf['audioFunc']['snr_threshold']
        self.chStr_use = conf['audioFunc']['chStr_use']  # channels to use, as in ['audio']['chStr']
        self.chMixMethod = conf['audioFunc']['chMixMethod']  # mean or max
        self.chStr = conf['audio']['chStr']
        self.bin_seconds = conf['audioFunc']['bin_seconds']
        self.nchannels = len(self.chStr)
        self.ch_idx = [i for i, ch in enumerate(self.chStr) if ch in self.chStr_use]
        if self.chMixMethod == 'mean':
            self.mmethod = np.mean
        elif self.chMixMethod == 'max':
            self.mmethod = np.max
        self.chunk_dur = conf['audio']['szChunk'] / self.sr  # duration of one streamed audio chunk
        self.n_chunks_perbin = int(np.round(self.bin_seconds / self.chunk_dur))
        self.bin_seconds_corrected = self.n_chunks_perbin * self.chunk_dur

        # plot init
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.ax2 = self.ax.twinx()

        self.line, = self.ax.plot([0], [0], lw=1, c="turquoise", marker='o', markersize=2,
                                  label=f"Click Count / {self.chunk_dur:.2f}s (subs.)")
        self.line1, = self.ax.plot([0], [0], lw=1, c="lightskyblue", marker='o', markersize=2,
                                   label=f"Click Count / {self.bin_seconds_corrected / 60:.2f} min")
        self.line2, = self.ax2.plot([None], [None], lw=2, marker='D', c='k', markersize=3, label="Cumulative (subs.)")
        self.lines = [self.line, self.line1, self.line2]

        self.ax.yaxis.label.set_color(self.line.get_color())
        self.ax.spines["left"].set_edgecolor(self.line.get_color())
        self.ax.legend(self.lines, [l.get_label() for l in self.lines], loc="upper left", fontsize=6)
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Click Count")

        self.ax2.set_yscale("log")
        self.ax2.set_ylim(1, 1.1e7)
        self.ax2.set_yticks([10 ** x for x in range(7)], minor=True)
        self.ax2.set_ylabel("Cumulative Click Count")
        plt.grid(True, which="both")
        self.ax.spines['top'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax.set_title(
            f"SNR estimate across 0 detected peaks: None", fontsize=8)
        self.fig.tight_layout()

        # counters
        self.cnt = 0
        self.i = 0
        self.t = []
        self.t_bin = []
        self.start_time = datetime.now()
        self.click_count_chunks = []
        self.click_count_binned = []
        self.click_count_cumulative = []
        self.snr_estimate = None

    def loop(self, q, qTrig, q_audio2triggered):
        try:
            # get data
            current_time = datetime.now()
            chunk = q.get(timeout=0.5)  # for now without timestamps
            data = chunk[:-1, :]  # remove trigger channel

            # reduce to one channel
            if self.nchannels > 1:
                data = self.mmethod(data, axis=0)
            else:
                data = data[0]

            # find peaks
            peak_frames, peak_heights, snr = self.find_peaks(data)
            self.t.append((current_time - self.start_time).total_seconds())
            self.click_count_chunks.append(len(peak_frames))
            self.click_count_cumulative.append(np.sum(np.array(self.click_count_chunks)))
            self.i += 1
            if self.i // self.n_chunks_perbin > len(self.click_count_binned):
                self.t_bin.append(np.array(self.t[-self.n_chunks_perbin:]).mean())
                self.click_count_binned.append(np.array(self.click_count_chunks[-self.n_chunks_perbin:]).sum())
                self.line1.set_data(np.array(self.t_bin) / 60, self.click_count_binned)

            # estimate snr
            if snr is not None:
                if self.cnt == 0:
                    self.snr_estimate = snr
                else:
                    self.snr_estimate = (self.cnt / (self.cnt + 1)) * self.snr_estimate + snr / (self.cnt + 1)
                self.cnt += 1

            # update the line plot
            max_n_points = 2000
            ss = int(
                max(1, len(self.click_count_chunks) // max_n_points))  # only show subset if many points are reached
            self.line.set_data(np.array(self.t)[::ss] / 60, self.click_count_chunks[::ss])
            self.line2.set_data(np.array(self.t)[::ss] / 60, self.click_count_cumulative[::ss])

            self.ax.relim()
            self.ax.autoscale_view()
            if self.snr_estimate is not None:
                self.ax.set_title(
                    f"SNR estimate across {self.click_count_cumulative[-1]} detected peaks: {self.snr_estimate:.1f}",
                    fontsize=8)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.tight_layout()

            # Can trigger the upon_trigger function in a module of the triggered folder here:
            #   qTrig.set()

        except Empty:
            pass

    def find_peaks(self, data):
        lowcut = self.lowcut
        highcut = self.highcut
        snr_threshold = self.snr_threshold
        fs = self.sr

        b, a = signal.butter(4, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
        filtered_signal = signal.filtfilt(b, a, data)
        analytic_signal = signal.hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        baseline = np.std(envelope)
        peak_frames, _ = signal.find_peaks(envelope, height=snr_threshold * baseline,
                                           distance=fs // 160)  # Adjust height & distance

        peak_heights = envelope[peak_frames]
        if len(peak_heights):
            snr = peak_heights.mean() / baseline
        else:
            snr = None
        return peak_frames, peak_heights, snr

    def close(self):
        plt.ioff()
        fn = self.conf['pSavBaseName'] + 'click_overview.pdf'
        plt.savefig(fn, dpi=600)
        print(f"Saved click overview to {fn}")
        plt.close('all')
