import scipy.io
import numpy as np


scale_sigma = 33 / np.sqrt(33) * 0.25
ss = np.subtract(np.arange(1, 33 + 1), np.ceil(33 / 2))
ys = np.exp(-0.5 * (np.power(ss, 2)) / scale_sigma ** 2)
ysf = np.fft.fft(ys)
lam = 1e-2
learning_rate = 0.025

# CarScale

######## erstes Frame ########
xs0 = scipy.io.loadmat('xs0.mat')
np_xsf = np.fft.fft(xs0['xs'])
xsf0_2 = scipy.io.loadmat('xsf0_2.mat')  # same

new_num = np.multiply(ysf, np.conj(np_xsf))
new_sf_num0 = scipy.io.loadmat('new_sf_num0.mat')  # same

new_den = np.sum(np.real(np.multiply(np_xsf, np.conj(np_xsf))), axis=0)
new_sf_den0 = scipy.io.loadmat('new_sf_den0.mat')  # same


######## zweites Frame ########
xs1 = scipy.io.loadmat('xs1.mat')
np_xsf1 = np.fft.fft(xs1['xs'])
xsf1 = scipy.io.loadmat('xsf1.mat')  # same

scale_response = np.divide(
                np.sum(np.multiply(new_num, np_xsf1), axis=0),
                (new_den + lam))
real_part = np.real(np.fft.ifftn(scale_response))
scale_response1 = scipy.io.loadmat('scale_response1.mat')  # same

# second selected sample
xs1_train = scipy.io.loadmat('xs1_train.mat')
np_xsf1_train = np.fft.fft(xs1_train['xs'])
xsf1_train = scipy.io.loadmat('xsf1_train.mat')  # same

new_sf_num1 = scipy.io.loadmat('new_sf_num1.mat')
new_num1 = np.multiply(ysf, np.conj(np_xsf1_train))  # same

new_sf_den1 = scipy.io.loadmat('new_sf_den1.mat')
new_den1 = np.sum(np.real(np.multiply(np_xsf1_train, np.conj(np_xsf1_train))), axis=0)  # same

# result of training
matlab_sf_num1 = scipy.io.loadmat('model_sf_num1.mat')
model_num1 = np.add(
    np.multiply((1 - learning_rate), new_num),
    np.multiply(learning_rate, new_num1))  # same

matlab_sf_den1 = scipy.io.loadmat('model_sf_den1.mat')
model_den1 = np.add(
    np.multiply((1 - learning_rate), new_den),
    np.multiply(learning_rate, new_den1))  # same


######## drittes Frame ########
xs2_test = scipy.io.loadmat('xs2_test.mat')
np_xs2_test = np.fft.fft(xs2_test['xs'])
xsf2_test = scipy.io.loadmat('xsf2_test.mat')  # same

scale_response2 = scipy.io.loadmat('scale_response2.mat')
np_scale_response2 = np.divide(
                np.sum(np.multiply(model_num1, np_xs2_test), axis=0),
                (model_den1 + lam))
real_part2 = np.real(np.fft.ifftn(np_scale_response2))  # same

xs2_train = scipy.io.loadmat('xs2_train.mat')
np_xsf2_train = np.fft.fft(xs2_train['xs'])
xsf2_train = scipy.io.loadmat('xsf2_train.mat')  # same

new_sf_num2 = scipy.io.loadmat('new_sf_num2.mat')
new_num2 = np.multiply(ysf, np.conj(np_xsf2_train))  # same

new_sf_den2 = scipy.io.loadmat('new_sf_den2.mat')
new_den2 = np.sum(np.real(np.multiply(np_xsf2_train, np.conj(np_xsf2_train))), axis=0)  # same

matlab_sf_num2 = scipy.io.loadmat('matlab_sf_num2.mat')
model_num2 = np.add(
    np.multiply((1 - learning_rate), model_num1),
    np.multiply(learning_rate, new_num2))  # same

matlab_sf_den2 = scipy.io.loadmat('matlab_sf_den2.mat')
model_den2 = np.add(
    np.multiply((1 - learning_rate), model_den1),
    np.multiply(learning_rate, new_den2))  # same





