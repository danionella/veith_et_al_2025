from functools import partial
import cv2
import numba
import numpy as np
import h5py
import pandas as pd

class Fragmenter:

    def __init__(self, frame_shape, threshold=0, area_lims=(500, 1500), crop_shape=(50, 150), crop_offset=(0, 0),
                 bgsub=None, mask=None, blob_file=None):
        self.threshold = threshold
        self.frame_shape = frame_shape
        self.area_lims = area_lims
        self.frags = dict(dur=[], coordinates=[])
        self.last_labeltofrag = np.zeros((1024), dtype=np.int32) - 1
        self.last_labelmap = np.zeros(frame_shape, dtype=np.int32)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.nfrags = 0
        self.bgsub = bgsub
        self.mask = mask
        self.nblobs = 0
        self.h5file = h5py.File(blob_file, 'w')
        self.h5file.create_dataset('frame_shape', data=frame_shape)
        self.h5file.create_dataset('blobs', (0, *crop_shape), chunks=(1, *crop_shape), maxshape=(None, *crop_shape),
                                   compression='lzf', dtype='uint8')
        self.decimation = 10
        self.bufsize = 100
        self.blobbuffer = np.zeros((self.bufsize, *crop_shape), dtype=np.uint8)
        self.blobinfobuffer = np.zeros((self.bufsize, 18), dtype='float32')
        self.dataframes = []

    def writebuffers(self, bufsz=None):
        if not bufsz: bufsz = self.bufsize
        if bufsz == 0: return
        dimsize = self.h5file['blobs'].shape[0] + bufsz
        self.h5file['blobs'].resize(dimsize, axis=0)
        self.h5file['blobs'][-bufsz:] = self.blobbuffer[:bufsz]
        self.dataframes.append(pd.DataFrame(self.blobinfobuffer[:bufsz].copy(),
                                            columns=('frag_idx', 'frame_idx', 'y', 'x', 'theta', 'w', 'l', 'area',
                                                     'M00', 'M01', 'M10', 'M11', 'M02', 'M20', 'M21', 'M12', 'M30',
                                                     'M03')))

    def step(self, frame, iFrame):
        assert (frame.dtype == np.uint8), "frame needs to be of np.uint8 data type"
        assert (frame.shape == self.frame_shape), "unexpected frame shape"
        ft = cv2.threshold(frame, self.threshold, 255, cv2.THRESH_BINARY)[1]
        ft = cv2.morphologyEx(ft, cv2.MORPH_OPEN, self.morph_kernel)
        if self.mask is not None:
            ft &= self.mask
        nlabels, labelmap = cv2.connectedComponentsWithAlgorithm(ft, connectivity=8, ccltype=cv2.CCL_GRANA,
                                                                 ltype=cv2.CV_32S)
        overlaps, moments = self.get_overlaps(self.last_labelmap, labelmap, frame)
        areas = np.sum(overlaps, axis=0)  # moments[:,0]
        nice_labels = np.where((areas[1:] < self.area_lims[1]) & (areas[1:] > self.area_lims[0]))[0] + 1
        labeltofrag = np.zeros((1024), dtype=np.int32) - 1

        if len(nice_labels > 0):
            blobs = {"centroid": [],
                     "theta": [],
                     "cropped": []}

            for iLabel in nice_labels:
                centroid = moments[iLabel, 2:0:-1] / moments[iLabel, 0]
                theta, w, l = self.get_ellipse(moments[iLabel, :])
                cropped = self.get_crop(frame, centroid, theta, crop_shape=self.crop_shape, offset=self.crop_offset)

                blobs["centroid"].append(centroid)
                blobs["theta"].append(theta)
                blobs["cropped"].append(cropped)

            last_label = np.where((overlaps[1:, iLabel] > 0) & (self.last_labeltofrag[1:] >= 0))[0] + 1
            num_overlaps = np.sum(overlaps[last_label][:, nice_labels] > 0)
            if num_overlaps == 1:  # one new blob overlaps with one previous blob
                labeltofrag[iLabel] = self.last_labeltofrag[last_label]
                self.frags['dur'][labeltofrag[iLabel]] += 1
            else:  # start a new fragment
                self.nfrags += 1
                labeltofrag[iLabel] = self.nfrags - 1
                self.frags['dur'].append(1)
                self.frags['coordinates'].append([])
            self.frags['coordinates'][labeltofrag[iLabel]].append(
                np.array([labeltofrag[iLabel], iFrame, centroid[0], centroid[1], theta, areas[iLabel]]))
            if (self.frags['dur'][labeltofrag[iLabel]] % self.decimation) == 1:  # if self.h5file:
                ikf = (self.nblobs % self.blobbuffer.shape[0])
                self.blobbuffer[ikf] = cropped.copy()
                self.nblobs += 1
                self.blobinfobuffer[ikf] = [int(labeltofrag[iLabel]), int(iFrame), centroid[0], centroid[1], theta,
                                            w, l, areas[iLabel], *moments[iLabel, :]]
                if (self.nblobs % self.blobbuffer.shape[0]) == 0:
                    self.writebuffers()
            lastfrags = self.last_labeltofrag[self.last_labeltofrag >= 0]
            unaccounted_frags = lastfrags[np.in1d(lastfrags, labeltofrag[labeltofrag >= 0], invert=True)]
            for frag in unaccounted_frags:
                self.frags['coordinates'][frag] = np.array(self.frags['coordinates'][frag])
            self.last_labeltofrag = labeltofrag
            self.last_labelmap = labelmap
        else:
            blobs = None

        return blobs


    def run_on_frame(self, frame, i):
        if self.bgsub is not None:
            if (i % 50) == 0:
                frame = self.bgsub.step(frame)
            else:
                frame = cv2.subtract(np.uint8(self.bgsub.bg), frame)
        blobs = self.step(frame, i)
        return frame, blobs

    def close(self):
        self.writebuffers(bufsz=(self.nblobs % self.blobbuffer.shape[0]))
        fn = self.h5file.filename
        self.h5file.close()
        try:
            pd.concat(self.dataframes, ignore_index=True).to_hdf(fn, key='blobinfo')
            pd.DataFrame({'frag_dur': np.asarray(self.frags['dur'])}).to_hdf(fn, key='frag_info')
            pd.DataFrame(np.concatenate(self.frags['coordinates'], axis=0).astype(np.float32),
                         columns=('frag_idx', 'frame_idx', 'y', 'x', 'theta', 'area')).to_hdf(fn, key='allcoords')
        except ValueError:
            print("No blobs detected")
            pass


    @staticmethod
    @numba.njit((numba.int32[:, ::1], numba.int32[:, ::1], numba.uint8[:, ::1]), parallel=False, fastmath=True,
                nogil=True)
    def get_overlaps(labels1, labels2, img):
        """Numba-accelerated function to calculate overlap matrix and image moments

        Args:
            labels1: label map as generated by cv2.connectedComponents
            labels2: second label map
            img: grayscale image

        Returns:
            2-element tuple containing
                - (numpy Array): overlap matrix
                - (numpy Array): image moments
        """
        overlaps = np.zeros((1024, 1024), dtype=np.int32)
        mom = np.zeros((1024, 10), dtype=np.float32)
        xmax = labels1.shape[0]
        ymax = labels1.shape[1]
        for xi in range(xmax):
            for yi in range(ymax):
                v = labels2[xi, yi]
                if v > 0:
                    u = labels1[xi, yi]
                    if v < 1024:
                        if u < 1024: overlaps[u, v] += 1
                        imgval = numba.float32(img[xi, yi])
                        x = numba.float32(xi)
                        y = numba.float32(yi)
                        mom[v, 0] += imgval
                        mom[v, 1] += y * imgval
                        mom[v, 2] += x * imgval
                        mom[v, 3] += x * y * imgval
                        mom[v, 4] += y * y * imgval
                        mom[v, 5] += x * x * imgval
                        mom[v, 6] += x * x * y * imgval
                        mom[v, 7] += x * y * y * imgval
                        mom[v, 8] += x * x * x * imgval
                        mom[v, 9] += y * y * y * imgval
        return overlaps, mom


    @staticmethod
    @numba.njit
    def get_ellipse(moments):
        # unpacks the list of moments generated by getOverlaps and calculates ellipse angle
        M00, M01, M10, M11, M02, M20, M21, M12, M30, M03 = moments[0:10]
        xbar = M10 / M00
        ybar = M01 / M00
        u11 = M11 - xbar * M01
        u20 = M20 - xbar * M10
        u02 = M02 - ybar * M01
        u21 = M21 - 2 * xbar * M11 - ybar * M20 + 2 * (xbar ** 2) * M01
        u12 = M12 - 2 * ybar * M11 - xbar * M02 + 2 * (ybar ** 2) * M10
        u30 = M30 - 3 * xbar * M20 + 2 * (xbar ** 2) * M10
        u03 = M03 - 3 * ybar * M02 + 2 * (ybar ** 2) * M01  # hu2 = 4*(u11**2)+(u20-u02)**2
        l = np.sqrt(8 * (u20 + u02 + np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2)))  # with np.errstate(invalid='ignore'):
        w = np.sqrt(8 * (u20 + u02 - np.sqrt(4 * u11 ** 2 + (u20 - u02) ** 2)))
        theta = 0.5 * np.arctan2(2 * u11, (u20 - u02))
        cost = np.cos(theta)
        sint = np.sin(theta)
        u30_rot = 1 * cost ** 3 * u30 + 3 * cost ** 2 * sint * u21 + 3 * cost * sint ** 2 * u12 + sint ** 3 * u03
        if u30_rot < 0: theta = np.mod(theta + np.pi, 2 * np.pi)
        return theta, w, l


    @staticmethod
    def get_crop(frame, centroid, theta=0, crop_shape=(200, 200), offset=(0, 0)):
        # rotates and an image around a point of interest
        M = cv2.getRotationMatrix2D((centroid[1], centroid[0]), 90 - theta * 180 / np.pi, 1)
        M[:, 2] -= centroid[1] - crop_shape[1] / 2 + offset[1], centroid[0] - crop_shape[0] / 2 + offset[0]
        cropped = cv2.warpAffine(frame, M, crop_shape[::-1])
        return cropped


class BgSub:

    def __init__(self, bg, bgfun=None, **kwargs):
        self.bg = bg
        self.lastframe = np.uint8(self.bg)
        if bgfun is None: bgfun = self.static_diff
        if kwargs is None:
            self.bgfun = bgfun
        else:
            self.bgfun = partial(bgfun, **kwargs)
        self.bgfun(self.bg, self.lastframe, self.lastframe)

    def step(self, frame):
        frame_bgsub = self.bgfun(self.bg, frame, self.lastframe)
        return frame_bgsub

    @staticmethod
    @numba.njit(['uint8[:,:](float32[:,:], uint8[:,:], uint8[:,:], float64, uint8)'], fastmath=True, nogil=True)
    def diffgated_average(bg, frame, last_frame, rate=0.001, diff_thresh=3):
        b = 1. - rate
        xmax, ymax = frame.shape
        new_frame = np.zeros(frame.shape, dtype=np.uint8)
        for x in range(xmax):
            for y in range(ymax):
                if bg[x, y] > frame[x, y]:
                    new_frame[x, y] = bg[x, y] - frame[x, y]
                if last_frame[x, y] <= frame[x, y]:
                    if (frame[x, y] - last_frame[x, y]) <= diff_thresh:
                        bg[x, y] = b * bg[x, y] + rate * frame[x, y]
                else:
                    if (last_frame[x, y] - frame[x, y]) <= diff_thresh:
                        bg[x, y] = b * bg[x, y] + rate * frame[x, y]
                last_frame[x, y] = frame[x, y]
        return new_frame

    @staticmethod
    @numba.njit(['uint8[:,::1](float32[:,::1], uint8[:,::1], uint8[:,::1], float64)'], fastmath=True, nogil=True)
    def running_average(bg, frame, last_frame=None, rate=0.001):
        b = 1. - rate
        xmax, ymax = frame.shape
        new_frame = np.zeros(frame.shape, dtype=frame.dtype)
        for x in range(xmax):
            for y in range(ymax):
                if bg[x, y] > frame[x, y]:
                    new_frame[x, y] = bg[x, y] - frame[x, y]
                bg[x, y] = b * bg[x, y] + rate * frame[x, y]
        return new_frame

    @staticmethod
    @numba.njit(['uint8[:,::1](uint8[:,::1], uint8[:,::1], uint8[:,::1])',
                 'uint8[:,::1](float32[:,::1], uint8[:,::1], uint8[:,::1])'], fastmath=True, nogil=True)
    def static_diff(bg, frame, last_frame=None):
        xmax, ymax = frame.shape
        new_frame = np.zeros(frame.shape, dtype=frame.dtype)
        for x in range(xmax):
            for y in range(ymax):
                if bg[x, y] > frame[x, y]:
                    new_frame[x, y] = bg[x, y] - frame[x, y]
        return new_frame
