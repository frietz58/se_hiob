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

        self.scale_window = None
        self.pos = None
        self.init_target_sz = None
        self.base_target_sz = None
        self.frame = None

    def setup(self, n_scales, scale_step, scale_sigma_factor, img_files, lam):

        self.nScales = n_scales
        self.scale_step = scale_step
        self.scale_sigma_factor = scale_sigma_factor
        self.img_files = img_files
        self.lam = lam

    def execute_scale_estimation(self, frame):

        # target size at scale = 1
        self.base_target_sz = frame.predicted_position

        sz = np.floor(self.base_target_sz * (1 + self.padding));

        # desired scale filter output (gaussian shaped), bandwidth proportional to
        # number of scales
        scale_sigma = self.nScales / np.sqrt(33) * self.scale_sigma_factor  # correct val
        ss = np.arange(-15, 17)  # correct val
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
        scale_model_factor = 1;
        if np.prod(self.init_target_sz) > self.scale_model_max_area:
            scale_model_factor = np.sqrt(self.scale_model_max_area / np.prod(self.init_target_sz))

        scale_model_sz = np.floor(self.init_target_sz * scale_model_factor)

        currentScaleFactor = 1

        # find maximum and minimum scales
        im = cv2.imread(self.img_files[1]);
        min_scale_factor = np.power(self.scale_step,
                                    np.rint(np.log(np.amax(np.divide(5, sz)) / np.log(self.scale_step))))
        max_scale_factor = np.power(self.scale_step,
                                    np.floor(np.log(np.amin(np.divide(
                                        [np.shape(im, 1),
                                         np.shape(im, 2)],
                                        self.base_target_sz))) / np.log(self.scale_step)))

        for frame in len(self.img_files):
            im = cv2.imread(self.img_files[frame])

            if frame > 1:

                # extract the test sample feature map for the scale filter
                xs = self.get_scale_sample(im, frame.predicted_position, self.base_target_sz, currentScaleFactor * scale_factors, scale_window,
                                      scale_model_sz)

                # calculate the correlation response of the scale filter
                xsf = np.fft.fftn(xs, [], 2);  # TODO make sure this is correct
                scale_response = np.real(np.fft.ifftn(np.divide(np.sum(np.multiply(sf_num, xsf), 1), (sf_den + self.lam))))

                # find the maximum scale response
                recovered_scale = np.nonzero(scale_response == np.amax(scale_response, 1))

                # update the scale
                currentScaleFactor = currentScaleFactor * scale_factors(recovered_scale)  # TODO
                if currentScaleFactor < min_scale_factor:
                    currentScaleFactor = min_scale_factor;
                elif currentScaleFactor > max_scale_factor:
                    currentScaleFactor = max_scale_factor;

                # extract the training sample feature map for the scale filter
                xs = self.get_scale_sample(im, frame.predicted_position, self.base_target_sz, currentScaleFactor * scale_factors, scale_window,
                                      scale_model_sz)

                # calculate the scale filter update
                xsf = np.fft.fftn(xs, [], 2);
                new_sf_num = np.multiply(ysf, np.conj(xsf))
                new_sf_den = np.sum(np.multiply(xsf, np.conj(xsf), 1))

                if frame == 1:
                    # first frame, train with a single image
                    sf_den = new_sf_den;
                    sf_num = new_sf_num;
                else:
                    # subsequent frames, update the model
                    sf_den = (1 - self.learning_rate) * sf_den + self.learning_rate * new_sf_den;
                    sf_num = (1 - self.learning_rate) * sf_num + self.learning_rate * new_sf_num;

    def get_scale_sample(self, im, pos, scaleFactors, scale_window, scale_model_sz):

        # Extracts a sample for the scale filter at the current
        # location and scale.

        nScales = len(scaleFactors)

        for s in len(nScales):
            patch_sz = np.floor(self.base_target_sz * scaleFactors[s])

            xs = np.floor(pos[2]) + np.arange(1, patch_sz[2] + 1) - np.floor(patch_sz[2] / 2);
            ys = np.floor(pos[1]) + np.arange(1, patch_sz[1] + 1) - np.floor(patch_sz[1] / 2);

            # check for out - of - bounds coordinates, and set them to the values at the borders
            if xs < 1:
                xs = 1

            if ys < 1:
                ys = 1

            if xs > np.shape(im)[2]:
                xs = np.shape(im)[2]

            if ys > np.shape(im)[1]:
                ys = np.shape(im)[1]

            # extract image
            im_patch = (im[ys, xs])  # TODO is this right?

            # resize image to model size
            im_patch_resized = cv2.resize(im_patch, scale_model_sz)

            # extract scale features
            winSize = (32, 32)
            blockSize = (16, 16)
            blockStride = (16, 16)
            cellSize = (4, 4)
            hog = cv2.HOGDescriptor(winSize, blockStride, blockSize, cellSize)

            temp_hog = hog(im_patch_resized)
            temp = temp_hog[:, :, 1: 31]  # ...TODO what?

            if s == 1:
                out = np.zeros(np.size(temp), nScales, 'single')

            #
            out[:, s] = np.multipy(temp[:, :], scale_window(s))

        return out

    def test(self):
        print("Module import works")



