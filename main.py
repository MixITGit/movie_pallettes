from vlc_player import Player
import sys
import os
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from sklearn.cluster import KMeans

CLUSTERS = 5
MARGIN = 5
BORDER_SIZE = 40
OFFSET = 2


class CustomVLCPlayer(Player):
    frames_taken = 0
    frames_taken_list = []
    frames_specified = 0

    def __init__(self):
        super(CustomVLCPlayer, self).__init__()

        self.videoframe.setFixedWidth(640)
        self.videoframe.setFixedHeight(360)

        self.snapbox = QtWidgets.QHBoxLayout()

        self.snapbutton = QtWidgets.QPushButton("Take Snapshot")
        self.snapbutton.setEnabled(False)
        self.snapbox.addWidget(self.snapbutton)
        self.snapbutton.clicked.connect(self.take_snapshot)
        #self.connect(self.snapbutton, QtCore.SIGNAL("clicked()"),
        #                     self.take_snapshot)

        self.l1 = QtWidgets.QLabel("Number of frames:")
        self.snapbox.addWidget(self.l1)

        self.sp = QtWidgets.QSpinBox()
        self.sp.setMaximum(10)
        self.sp.setMinimum(0)
        self.snapbox.addWidget(self.sp)
        self.sp.valueChanged.connect(self.valuechange)

        self.snapbox.addStretch(1)

        self.l2 = QtWidgets.QLabel("Frames taken: "+str(self.frames_taken))
        self.l2.setFixedHeight(24)
        self.snapbox.addWidget(self.l2)
        self.l2.setVisible(False)

        self.vboxlayout.addLayout(self.snapbox)

        self.imageareaWidget = QtWidgets.QWidget(self)
        self.imageareaWidget.setFixedHeight(80)

        self.imagearea = QtWidgets.QHBoxLayout(self.imageareaWidget)

        self.imageBoxes = []
        for i in range(0, 10):
            self.imageBoxes.append(QtWidgets.QLabel(str(i)))
            self.imagearea.addWidget(self.imageBoxes[len(self.imageBoxes)-1])

        self.vboxlayout.addWidget(self.imageareaWidget)
        self.imageareaWidget.setVisible(False)

    def valuechange(self):

        self.frames_specified = self.sp.value()

        self.l2.setText("Frames taken: "+str(self.frames_taken) +
                        " from "+str(self.frames_specified))

        self.sp.setEnabled(False) if (self.frames_taken > 0) else self.sp.setEnabled(True)
        if (self.sp.value() > 0
                and self.frames_taken < 10
                and self.frames_taken < self.frames_specified):
            self.snapbutton.setEnabled(True)
        else:
            self.snapbutton.setEnabled(False)

        if self.sp.value() > 0:
            self.l2.setVisible(True)
        else:
            self.l2.setVisible(False)

    def take_snapshot(self):

        wasPlaying = None

        videoSize = self.mediaplayer.video_get_size()

        self.mediaplayer.video_take_snapshot(
            0, "./img_"+str(self.frames_taken)+".png",
            videoSize[0],
            videoSize[1])

        if self.mediaplayer.is_playing():
            self.PlayPause()
            wasPlaying = True

        imagePath = os.getcwd() + "/img_" + str(self.frames_taken)+".png"

        image = cv2.imread(imagePath)

        image_copy = image_resize(cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB), width=100)

        pixelImage = image_copy.reshape(
            (image_copy.shape[0] * image_copy.shape[1], 3))

        clt = KMeans(n_clusters=CLUSTERS + OFFSET)
        clt.fit(pixelImage)

        hist = centroid_histogram(clt)

        bar = plot_colors(hist, clt.cluster_centers_)

        barImage = image_resize(
            cv2.cvtColor(bar, cv2.COLOR_RGB2BGR),
            width=int(videoSize[0]))

        im = np.zeros((BORDER_SIZE // 2, int(videoSize[0]), 3), np.uint8)
        cv2.rectangle(im, (0, 0), (int(videoSize[0]), BORDER_SIZE // 2),
                      (255, 255, 255), -1)

        newImg = np.concatenate([image, im, barImage], axis=0)
        cv2.imwrite(imagePath, newImg)

        pixmap = QtGui.QPixmap()
        pixmap.load(imagePath)
        pixmap = pixmap.scaledToWidth(50)
        self.imageBoxes[self.frames_taken].setPixmap(pixmap)
        self.frames_taken = self.frames_taken + 1

        self.imageareaWidget.setVisible(True)
        self.valuechange()

        self.frames_taken_list.append(imagePath)

        if self.frames_taken == self.frames_specified:
            resultImgs = []
            for i in self.frames_taken_list:
                # here we just add a whitespace rectangle as a top margin
                resultImgs.append(cv2.imread(i))
                im = np.zeros((
                    BORDER_SIZE * 2,
                    cv2.imread(i).shape[1],
                    3), np.uint8)
                cv2.rectangle(
                    im,
                    (0, 0),
                    (cv2.imread(i).shape[1], BORDER_SIZE * 2),
                    (255, 255, 255), -1)
                # here we just add a whitespace rectangle as a bottom margin
                resultImgs.append(im)

            # Now that we have the list of all the needed images,
            # we concatinate them vertically and form the final image
            final = np.concatenate(resultImgs, axis=0)

            # The final image still needs an outside border, so the last task
            # is to create this border line
            finalShape = final.shape
            w = finalShape[1]
            h = finalShape[0]
            base_size = h + BORDER_SIZE, w + BORDER_SIZE, 3
            base = np.zeros(base_size, dtype=np.uint8)
            # We combine our main image with the border line
            cv2.rectangle(
                base,
                (0, 0),
                (w + BORDER_SIZE, h + BORDER_SIZE),
                (255, 255, 255), BORDER_SIZE)
            base[
            (BORDER_SIZE / 2):h + (BORDER_SIZE / 2),
            (BORDER_SIZE / 2):w + (BORDER_SIZE / 2)
            ] = final

            # The final output image is ready.
            # We now export it to the root directory
            cv2.imwrite("result_image.png", base)
            sys.exit(app.exec_())

            # If the image capture process continues, we resume playing the video
        self.PlayPause() if wasPlaying else None

# Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


# Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # Sort the centroids to form a gradient color look
    centroids = sorted(centroids, key=lambda x: sum(x))

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids[OFFSET:]):
        # plot the relative percentage of each cluster
        # endX = startX + (percent * 300)

        # Instead of plotting the relative percentage,
        # we will make a n=clusters number of color rectangles
        # we will also seperate them by a margin
        new_length = 300 - MARGIN * (CLUSTERS - 1)
        endX = startX + new_length/CLUSTERS
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        cv2.rectangle(bar, (int(endX), 0), (int(endX + MARGIN), 50),
                      (255, 255, 255), -1)
        startX = endX + MARGIN

    # return the bar chart
    return bar


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized



app = QtWidgets.QApplication(sys.argv)
player = CustomVLCPlayer()
player.show()
player.resize(660, 530)
sys.exit(app.exec_())