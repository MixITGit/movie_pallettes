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

        im = np.zeros((BORDER_SIZE / 2, int(videoSize[0]), 3), np.uint8)
        cv2.rectangle(im, (0, 0), (int(videoSize[0]), BORDER_SIZE / 2),
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