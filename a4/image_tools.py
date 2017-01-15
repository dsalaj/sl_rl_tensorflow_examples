import numpy as np
from scipy import misc

def get_all_coords(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1e
    opponent_slice = I[:, 9:10]
    opponent_x = 10.
    opponent_y = ((np.argmax(opponent_slice, axis=0) + (
        opponent_slice.shape[0] - np.argmax(np.array(list(reversed(opponent_slice))), axis=0))) / 2)[0]
    self_slice = I[:, 70:71]
    self_x = 70.
    self_y = ((np.argmax(self_slice, axis=0) + (
        self_slice.shape[0] - np.argmax(np.array(list(reversed(self_slice))), axis=0))) / 2)[0]
    ball_slice = I[:, 10:70]
    ball_x = 10. + np.argmax(np.sum(ball_slice, axis=0))
    ball_y = np.argmax(np.sum(ball_slice, axis=1))
    return float(opponent_y - self_y), float(ball_x - self_x), float(ball_y - self_y)
    # return self_x, self_y, opponent_x, opponent_y, ball_x, ball_y



def rgb2gray(I):
    return np.dot(I[...,:3], [0.299, 0.587, 0.114])

def crop(I):
    I_p = rgb2gray(I)
    I_p = misc.imresize(I_p, (100,84))
    I_p = I_p[13:97]
    I_p = I_p / 255.
    return I_p

# downsampling
def prepro(I1, I2, I3, I4):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I1_p = crop(I1)
    I2_p = crop(I2)
    I3_p = crop(I3)
    I4_p = crop(I4)

    return np.stack((I1_p, I2_p, I3_p, I4_p), axis = 2)  #np.array([I1_p,I2_p,I3_p,I4_p])

def prepro_14(I1, I4):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I1_p = crop(I1)
    I4_p = crop(I4)

    return np.stack((I1_p, I4_p), axis = 2).astype(np.float16)