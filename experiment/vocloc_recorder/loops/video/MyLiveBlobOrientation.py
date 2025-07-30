import cv2
from queue import Empty

# MyLiveBlob
import numpy as np
from loops.utils.videoprocessing import track_live
import pandas as pd


class MyLiveBlobOrientation:
    """Live fish detection
    Fish are detected as moving blobs. When the fish swims into a trigger zone, the trigger event is set.
    """

    def __init__(self, conf):
        self.conf = conf
        self.sc = conf["videoFunc"]["sub_spatial"]
        self.height = conf["camera"]["height"] // self.sc
        self.width = conf["camera"]["width"] // self.sc
        self.im_diff = np.zeros((self.height, self.width, 3))

        # Preview of fish position
        self.showblobs = 1
        if self.showblobs:
            cv2.namedWindow('MyLiveBlob', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MyLiveBlob', conf['vis']['camWindowSize'][0], conf['vis']['camWindowSize'][1])
            cv2.moveWindow('MyLiveBlob', 46 * 30, 1 * 30)
            cv2.waitKey(1)

        # Preview of fish in first blob, fixed orientation
        self.showfish = 1
        if self.showfish:
            cv2.namedWindow('Fish', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Fish', 512, 128)
            cv2.moveWindow('Fish', 46 * 30, 14 * 30)
            cv2.waitKey(1)

        # Background subtraction param
        self.nframes = 5  # costly median calculation
        self.every_n = 1  # after temporal sub-sampling
        self.cnt = 0
        min_detection_delay = self.nframes * self.every_n * self.conf["videoFunc"]["sub_t"]
        custom_detection_delay = 2 * 60 * conf["trigger"]["rate"]  # 2 * 60 * conf["trigger"]["rate"]
        self.start_detection = max(min_detection_delay, custom_detection_delay)
        self.track_init_flag = False
        print(f"Start detection after {self.start_detection / conf['trigger']['rate']:.1f}s.")
        self.frame_store = np.zeros((self.nframes, self.height, self.height), dtype=np.uint8)
        self.threshold = 4
        self.lower_area_lim = 53

        # Trigger area
        self.ch_w = np.pi / 4  # orientation constraint, cone around vertical axis: half width
        # size and shape is computed on the fly after definition of calibration area

        # Trigger start
        self.start_triggering = 5 * 60 * conf["trigger"]["rate"]  # 5 * 60 * conf["trigger"]["rate"]
        print(f"Start triggering after {self.start_triggering / conf['trigger']['rate']:.1f}s.")

        # Image counter
        self.i = -1

        # Mouse Callback
        self.callback_set = 0
        self.callback_finished = 0
        self.ys = []
        self.xs = []
        self.miny = None
        self.minx = None
        self.gridh = None
        self.gridw = None

    def draw_circle(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.im, (x, y), 4, (255, 0, 0), -1)
            self.ys.append(y)
            self.xs.append(x)
        if len(self.ys) == 4 and not self.callback_finished:
            self.miny = min(self.ys)
            self.minx = min(self.xs)
            self.gridh = max(self.ys) - min(self.ys)
            self.gridw = max(self.xs) - min(self.xs)
            self.grid_centeryx = [self.miny + self.gridh // 2, self.minx + self.gridw // 2]
            self.hw = min(self.gridh, self.gridw) // 4
            self.hh = min(self.gridh, self.gridw) // 8
            self.callback_finished = 1
            print("All points set.")
            print(f"y_min={self.miny}, x_min={self.minx}, gridh={self.gridh}, gridw={self.gridw}")

    def loop(self, q, qTrig, q_video2triggered):
        try:
            # Get frame
            t, idx, im = q.get(timeout=0.5)
            self.i = idx

            if not self.callback_set:
                self.callback_set = 1
                print("Double click to the four corners of the stimulation calibration area.")
                self.im = im.copy()
                cv2.namedWindow('StimsetBoundaries', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('MyLiveBlob', self.conf['vis']['camWindowSize'][0],
                                 self.conf['vis']['camWindowSize'][1])
                cv2.moveWindow('StimsetBoundaries', 29 * 30, 1 * 30)
                cv2.setMouseCallback('StimsetBoundaries', self.draw_circle)

            if self.callback_finished:
                self.roi = self.rectangle(centeryx=self.grid_centeryx, hw=self.hw, hh=self.hh)
                self.roi_BGR = np.dstack([255 * self.roi, 153 * self.roi, 51 * self.roi])

            # Collect frames for background subtraction
            if self.i * self.conf["videoFunc"]["sub_t"] < self.start_detection:
                if self.i * self.conf["videoFunc"]["sub_t"] > self.start_detection - (
                        self.nframes * self.every_n * self.conf["videoFunc"]["sub_t"]):
                    if self.i % self.every_n == 0:
                        self.frame_store[self.cnt] = im
                        self.cnt += 1

            # Init Detection
            if self.i * self.conf["videoFunc"]["sub_t"] >= self.start_detection:
                if not self.track_init_flag:
                    self.track_init_flag = True
                    print(f"Compute background from {len(self.frame_store)} images...")
                    self.bg = np.float32(np.median(self.frame_store, axis=0))
                    self.bgsub = track_live.BgSub(self.bg, bgfun=track_live.BgSub.diffgated_average, rate=1 / 100,
                                                  diff_thresh=3)
                    self.tr = track_live.Fragmenter(im.shape, threshold=self.threshold,
                                                    area_lims=(
                                                        self.lower_area_lim // self.sc ** 2, 1e12 // self.sc ** 2),
                                                    crop_shape=(32 // self.sc, 128 // self.sc),
                                                    crop_offset=(0 // self.sc, 20 // self.sc), bgsub=self.bgsub,
                                                    mask=None,
                                                    blob_file=self.conf['pSavVideo'] + '.blobs_sub_t' + '.h5')
                    print("Found background. Start Detection...")

            # Detect
            if self.i * self.conf["videoFunc"]["sub_t"] >= self.start_detection:
                im_diff, blobs = self.tr.run_on_frame(im, self.conf["videoFunc"]["sub_t"] * self.i)

                if self.showblobs:
                    self.im_diff = cv2.cvtColor(im_diff, cv2.COLOR_GRAY2BGR)
                    self.im_diff = 0.8 * self.im_diff + 0.1 * self.roi_BGR + 0.1 * cv2.cvtColor(im,
                                                                                                cv2.COLOR_GRAY2BGR)

                if blobs is not None:
                    # Trigger whatever you put into the triggered function
                    if self.i * self.conf["videoFunc"]["sub_t"] >= self.start_triggering:
                        xy_local = tuple(np.int64(blobs["centroid"][0][1::-1]))  # origin top left
                        if self.roi[xy_local[::-1]]:
                            # only trigger if the fish is roughly along the vertical axis
                            th = blobs["theta"][0]  # [-.5pi,1.5pi], up = 0 pi
                            th = (th + 0.5 * np.pi) % np.pi - 0.5 * np.pi  # up equals down, [-.5pi,.5pi]
                            if abs(th) < self.ch_w:
                                if not qTrig.is_set():
                                    content = np.array(
                                        [xy_local[0], xy_local[1], self.minx, self.miny, self.gridw, self.gridh])
                                    print(f"Video_func: {content}")
                                    q_video2triggered.clear()  # make sure no old position is waiting here
                                    q_video2triggered.put(content)
                                    qTrig.set()  # TRIGGER
                                    cv2.putText(self.im_diff, 'Triggered!', (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                .5,
                                                (255, 255, 255), 1, 2)
                            else:
                                if not qTrig.is_set():
                                    cv2.putText(self.im_diff, 'Fish not vertical.', (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                .5,
                                                (255, 255, 255), 1, 2)
                            if qTrig.is_set():
                                cv2.putText(self.im_diff, 'Trigger still set.', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                            (255, 255, 255), 1, 2)

                    # Preview
                    if self.showfish:
                        im_fish = blobs["cropped"][0]
                        cv2.imshow('Fish', im_fish.astype('uint8'))
                        cv2.waitKey(1)

                    if self.showblobs:
                        for i, yx in enumerate(blobs["centroid"]):
                            if i == 0:
                                color = (0, 153, 0)
                            else:
                                color = (51, 51, 255)
                            r_from_area = np.maximum(1, int(np.sqrt(self.lower_area_lim / np.pi // self.sc)))
                            self.im_diff = cv2.circle(self.im_diff, tuple(np.int64(yx[1::-1])),
                                                      color=color, radius=r_from_area, thickness=2)

                if self.showblobs:
                    cv2.imshow('MyLiveBlob', self.im_diff.astype('uint8'))

            cv2.imshow('StimsetBoundaries', self.im.astype('uint8'))
            cv2.waitKey(1)

            # test: deleteme
            # self.start_triggering = 5 * self.conf["trigger"]["rate"]
            # if self.i * self.conf["videoFunc"]["sub_t"] >= self.start_triggering:
            #    if not qTrig.is_set():
            #        qTrig.set()

        except Empty:
            pass

    def center_circle(self, r):
        """
        Draws a circle in the center of an image
        :param r: radius of the circle
        :return: binary mask, True inside circle
        """
        center = np.array([int(self.height / 2), int(self.width / 2)])[..., np.newaxis, np.newaxis]
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[np.linalg.norm(np.indices((self.height, self.width)) - center, axis=0) < r] = 1
        return mask

    def center_square(self, hw):
        """
        :param hw: Half width of square
        :return:
        """
        cy = int(self.height / 2)
        cx = int(self.width / 2)

        # mask = np.zeros((self.height, self.width), dtype=bool)
        # mask[cy-hw:cy+hw,cx-hw:cx+hw] = 1

        mask = np.zeros((self.height, self.width), dtype=np.bool)
        mask[cy - hw:cy + hw, cx - hw:cx + hw] = 1
        return mask

    def square(self, centeryx=[100, 100], hw=10):
        """
        :param centeryx:
        :param hw: half width
        :return:
        """
        return self.rectangle(centeryx=[100, 100], hh=hw, hw=hw)

    def rectangle(self, centeryx=[100, 100], hh=10, hw=10):
        """
        :param centeryx:
        :param hh: half height
        :param hw: half width
        :return:
        """
        cy = centeryx[0]
        cx = centeryx[1]
        mask = np.zeros((self.height, self.width), dtype=np.bool)
        mask[cy - hh:cy + hh, cx - hw:cx + hw] = 1

        return mask

    def close(self):
        self.tr.close()
        if self.conf["videoFunc"]["allowSaving"]:
            cv2.imwrite(self.conf['pSavBaseName'] + 'trigger_area.png', self.im_diff)
        cv2.destroyAllWindows()

        fn = self.conf['pSavBaseName'] + 'trigger_setting.csv'
        if self.conf["videoFunc"]["allowSaving"]:
            pd.DataFrame({'halfheight': self.hh,
                          'halfwidth': self.hw,
                          'start_triggering_abs_frame': self.start_triggering,
                          'blob_threshold': self.threshold,
                          'ch_w': self.ch_w,
                          'blob_lower_area_lim': self.lower_area_lim,
                          'grid_min_x': self.minx,
                          'grid_min_y': self.miny,
                          'grid_centery': self.grid_centeryx[0],
                          'grid_centerx': self.grid_centeryx[1],
                          'gridh': self.gridh,
                          'gridw': self.gridw
                          }, index=[0]).to_csv(fn)
