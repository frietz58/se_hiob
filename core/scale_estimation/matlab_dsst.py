import numpy as np
import cv2


class DsstEstimator:

    def __init__(self):

        self.padding = 1.0
        self.nScales = None
        self.scale_step = None
        self.scale_sigma_factor = None
        self.scale_model_max_area = None
        self.target_sz = None
        self.img_files = None
        self.lam = None
        self.learning_rate = None

        self.scale_window = None
        self.pos = None
        self.init_target_sz = None
        self.base_target_sz = None
        self.frame = None
        self.sf_den = None
        self.sf_num = None

    def setup(self, n_scales, scale_step, scale_sigma_factor, img_files, scale_model_max, learning_rate):

        self.nScales = n_scales
        self.scale_step = scale_step
        self.scale_sigma_factor = scale_sigma_factor
        self.img_files = img_files
        self.lam = 1e-2
        self.scale_model_max_area = scale_model_max
        self.learning_rate = learning_rate

    def execute_scale_estimation(self, frame):

        self.frame = frame

        # target size at scale = 1
        self.base_target_sz = [frame.predicted_position.width, frame.predicted_position.height]
        self.init_target_sz = [frame.predicted_position.width, frame.predicted_position.height]

        sz = np.floor(np.multiply(self.base_target_sz, (1 + self.padding)))

        # desired scale filter output (gaussian shaped), bandwidth proportional to
        # number of scales
        scale_sigma = self.nScales / np.sqrt(33) * self.scale_sigma_factor  # correct val
        ss = np.arange(-15, 18)  # not correct val, changed 17 to 18  for testing
        ys = np.exp(-0.5 * (np.power(ss, 2) / scale_sigma ** 2))  # correct val
        ysf = np.fft.fft(ys)  # correct val

        # store pre-computed scale filter cosine window
        # if mod(nScales,2) == 0
        if np.mod(self.nScales, 2) == 0:
            scale_window = np.hanning(self.nScales + 1)
            scale_window = scale_window[2: len(scale_window)]
        else:
            scale_window = np.hanning(self.nScales)  # correct val

        # scale factors
        ss = np.arange(1, self.nScales + 1)
        scale_factors = np.power(self.scale_step, np.subtract(np.rint(self.nScales / 2), ss))  # almost correct val? round error?

        # compute the resize dimensions used for feature extraction in the scale
        # estimation
        scale_model_factor = 1
        if np.prod(self.init_target_sz) > self.scale_model_max_area:
            scale_model_factor = np.sqrt(self.scale_model_max_area / np.prod(self.init_target_sz))

        scale_model_sz = np.floor(np.multiply(self.init_target_sz, scale_model_factor))

        currentScaleFactor = 1

        # find maximum and minimum scales
        im = self.img_files[1]
        min_scale_factor = np.power(self.scale_step,
                                    np.rint(np.log(np.amax(np.divide(5, sz)) / np.log(self.scale_step))))
        max_scale_factor = np.power(self.scale_step,
                                    np.floor(np.log(np.amin(np.divide(
                                        [np.shape(im)[0],
                                         np.shape(im)[1]],
                                        self.base_target_sz))) / np.log(self.scale_step)))

        im = self.img_files[frame.number]

        if self.frame.number > 1:

            # extract the test sample feature map for the scale filter
            xs = self.get_scale_sample(im, frame.predicted_position, self.base_target_sz,
                                       currentScaleFactor * scale_factors, scale_window,
                                       scale_model_sz)

            # calculate the correlation response of the scale filter
            xsf = np.fft.fft2(xs)
            scale_response = np.real(np.fft.ifftn(np.divide(np.sum(np.multiply(self.sf_num, xsf), 1),
                                                            (self.sf_den + self.lam))))

            # find the maximum scale response
            # recovered_scale = np.nonzero(scale_response == np.amax(scale_response))
            recovered_scale = np.where(scale_response == np.amax(scale_response))[0][0]

            # update the scale
            # currentScaleFactor = currentScaleFactor * scale_factors[recovered_scale]  # TODO
            currentScaleFactor = currentScaleFactor * scale_response[recovered_scale] # TODO is this right?
            if currentScaleFactor < min_scale_factor:
                currentScaleFactor = min_scale_factor
            elif currentScaleFactor > max_scale_factor:
                currentScaleFactor = max_scale_factor

        # extract the training sample feature map for the scale filter
        xs = self.get_scale_sample(im, frame.predicted_position, self.base_target_sz,
                                    currentScaleFactor * scale_factors, scale_window,
                                    scale_model_sz)

        # calculate the scale filter update
        xsf = np.fft.fft2(xs)
        new_sf_num = np.multiply(ysf, np.conj(xsf))
        new_sf_den = np.sum(np.multiply(xsf, np.conj(xsf)), 1)

        if self.frame.number == 1:
            # first frame, train with a single image
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            # subsequent frames, update the model
            self.sf_den = (1 - self.learning_rate) * self.sf_den + self.learning_rate * new_sf_den
            self.sf_num = (1 - self.learning_rate) * self.sf_num + self.learning_rate * new_sf_num

        target_sz = np.floor(np.multiply(self.base_target_sz, currentScaleFactor))

        return target_sz

    def get_scale_sample(self, im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz):

        # Extracts a sample for the scale filter at the current
        # location and scale.

        nScales = len(scaleFactors)
        first = True

        for s in range(nScales):
            patch_sz = np.floor(np.multiply(base_target_sz, scaleFactors[s]))

            #xs = np.floor(pos.x) + np.arange(1, patch_sz[1] + 1) - np.floor(patch_sz[1] / 2)
            #ys = np.floor(pos.y) + np.arange(1, patch_sz[0] + 1) - np.floor(patch_sz[0] / 2)

            xs = np.floor(pos.x) + patch_sz[1] + 1 - np.floor(patch_sz[1] / 2)
            ys = np.floor(pos.y) + patch_sz[0] + 1 - np.floor(patch_sz[0] / 2)

            # check for out - of - bounds coordinates, and set them to the values at the borders
            if xs < 1:
                xs = 1

            if ys < 1:
                ys = 1

            if xs > np.shape(im)[0]:
                xs = np.shape(im)[0]

            if ys > np.shape(im)[1]:
                ys = np.shape(im)[1]

            # extract image
            # im_patch = (im[ys, xs])  # TODO is this right?
            im_patch = im[
                       int(ys - np.floor(patch_sz[0] / 2)):
                       int(ys - np.floor(patch_sz[0] / 2) + patch_sz[0]),
                       int(xs - np.floor(patch_sz[1] / 2)):
                       int(xs - np.floor(patch_sz[1] / 2) + patch_sz[1])
                       ]

            # resize image to model size
            # im_patch_resized = cv2.resize(im_patch, scale_model_sz)
            # im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]), int(scale_model_sz[1])))
            im_patch_resized = cv2.resize(im_patch, (32, 32))

            # extract scale features
            # winSize = (int(scale_model_sz[0]), int(scale_model_sz[1]))
            winSize = (32, 32)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (4, 4)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            #hog = cv2.HOGDescriptor()
            temp_hog = hog.compute(im_patch_resized)
            # temp = temp_hog[:, :, 1: 31]  # ...TODO what?

            if first:
                out = np.zeros((np.size(temp_hog), nScales))
                first = False

            out[:, s] = np.multiply(temp_hog.flatten(), scale_window[s])

        # out becomes a matrix, where each column is the hog feature vector at a different scale lvl
        return out

    def test(self):
        print("Module import works")



