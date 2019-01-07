import numpy as np


class Eco:

    def __init__(self):

        self.init_target_sz = None
        self.n_scales = None
        self.scale_step = None
        self.scale_sigma = None
        self.scale_exp = None
        self.scale_exp_shift =None

    def init_scale_filter(self, number_scales, scale_step, scale_factors, scale_filter, params):

        self.init_target_sz = params.init_sz

        self.n_scales = number_scales
        self.scale_step = scale_step

        self.scale_sigma = params.number_of_interp_scales * params.scale_sigma_facgtor

        self.scale_exp = np.multiply(
            [-np.floor((self.n_scales - 1) / 2), np.ceil((self.n_scales - 1)/2)],
            params.number_of_interp_scales / self.n_scales)

        self.scale_exp_shift = np.roll(self.scale_exp, [0 - np.floor((self.n_scales - 1) / 2)])






