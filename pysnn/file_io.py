import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import torch


"""
Adapted from https://github.com/bamsumit/slayerPytorch
"""


#########################################################
# Event base class
#########################################################
class Events:
    r"""
    This class provides a way to store, read, write and visualize spike Events.

    Members:
        * ``x`` (numpy ``int`` array): `x` index of spike Events.
        * ``y`` (numpy ``int`` array): `y` index of spike Events (not used if the spatial dimension is 1).
        * ``p`` (numpy ``int`` array): `polarity` or `channel` index of spike Events.
        * ``t`` (numpy ``double`` array): `timestamp` of spike Events. Time is assumend to be in ms.

    Usage:

    >>> TD = file_io.Events(x_events, y_events, p_events, t_events)
    """

    def __init__(self, x_events, y_events, p_events, t_events):
        if y_events is None:
            self.dim = 1
        else:
            self.dim = 2

        self.x = (
            x_events if type(x_events) is np.array else np.asarray(x_events)
        )  # x spatial dimension
        self.y = (
            y_events if type(y_events) is np.array else np.asarray(y_events)
        )  # y spatial dimension
        self.p = (
            p_events if type(p_events) is np.array else np.asarray(p_events)
        )  # spike polarity
        self.t = (
            t_events if type(t_events) is np.array else np.asarray(t_events)
        )  # time stamp in ms

        self.p -= self.p.min()

    def to_spike_array(self, sampling_time=1, dim=None):
        r"""
        Returns a numpy tensor that contains the spike Eventss sampled in bins of `sampling_time`.
        The array is of dimension (channels, height, time) or``CHT`` for 1D data.
        The array is of dimension (channels, height, width, time) or``CHWT`` for 2D data.

        Arguments:
            * ``sampling_time``: the width of time bin to use.
            * ``dim``: the dimension of the desired tensor. Assignes dimension itself if not provided.
            y
        Usage:

        >>> spike = TD.to_spike_array()
        """
        if self.dim == 1:
            if dim is None:
                dim = (
                    np.round(max(self.p) + 1).astype(int),
                    np.round(max(self.x) + 1).astype(int),
                    np.round(max(self.t) / sampling_time + 1).astype(int),
                )
            frame = np.zeros((dim[0], 1, dim[1], dim[2]))
        elif self.dim == 2:
            if dim is None:
                dim = (
                    np.round(max(self.p) + 1).astype(int),
                    np.round(max(self.y) + 1).astype(int),
                    np.round(max(self.x) + 1).astype(int),
                    np.round(max(self.t) / sampling_time + 1).astype(int),
                )
            frame = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        return self.to_spike_tensor(frame, sampling_time).reshape(dim)

    def to_spike_tensor(self, empty_tensor, sampling_time=1):
        r"""
        Returns a numpy tensor that contains the spike Eventss sampled in bins of `sampling_time`.
        The tensor is of dimension (channels, height, width, time) or``CHWT``.

        Arguments:
            * ``empty_tensor`` (``numpy or torch tensor``): an empty tensor to hold spike data 
            * ``sampling_time``: the width of time bin to use.

        Usage:

        >>> spike = TD.to_spike_tensor( torch.zeros((2, 240, 180, 5000)) )
        """
        if self.dim == 1:
            x_events = np.round(self.x).astype(int)
            p_events = np.round(self.p).astype(int)
            t_events = np.round(self.t / sampling_time).astype(int)
            valid_ind = np.argwhere(
                (x_events < empty_tensor.shape[2])
                & (p_events < empty_tensor.shape[0])
                & (t_events < empty_tensor.shape[3])
            )
            empty_tensor[
                p_events[valid_ind], 0, x_events[valid_ind], t_events[valid_ind]
            ] = (1 / sampling_time)
        elif self.dim == 2:
            x_events = np.round(self.x).astype(int)
            y_events = np.round(self.y).astype(int)
            p_events = np.round(self.p).astype(int)
            t_events = np.round(self.t / sampling_time).astype(int)
            valid_ind = np.argwhere(
                (x_events < empty_tensor.shape[2])
                & (y_events < empty_tensor.shape[1])
                & (p_events < empty_tensor.shape[0])
                & (t_events < empty_tensor.shape[3])
            )
            empty_tensor[
                p_events[valid_ind],
                y_events[valid_ind],
                x_events[valid_ind],
                t_events[valid_ind],
            ] = (1 / sampling_time)
        return empty_tensor


#########################################################
# Conversion
#########################################################
def spike_array_to_events(spike_mat, sampling_time=1):
    r"""
    Returns TD Events from a numpy array (of dimension 3 or 4).
    The numpy array must be of dimension (channels, height, time) or``CHT`` for 1D data.
    The numpy array must be of dimension (channels, height, width, time) or``CHWT`` for 2D data.

    Arguments:
        * ``spike_mat``: numpy array with spike information.
        * ``sampling_time``: time width of each time bin.

    Usage:

    >>> TD = file_io.spike_array_to_events(spike)
    """
    if spike_mat.ndim == 3:
        spike_events = np.argwhere(spike_mat > 0)
        x_events = spike_events[:, 1]
        y_events = None
        p_events = spike_events[:, 0]
        t_events = spike_events[:, 2]
    elif spike_mat.ndim == 4:
        spike_events = np.argwhere(spike_mat > 0)
        x_events = spike_events[:, 2]
        y_events = spike_events[:, 1]
        p_events = spike_events[:, 0]
        t_events = spike_events[:, 3]
    else:
        raise Exception(
            "Expected numpy array of 3 or 4 dimension. It was {}".format(spike_mat.ndim)
        )

    return Events(x_events, y_events, p_events, t_events * sampling_time)


#########################################################
# 1D reading and encoding
#########################################################
def read_1d_spikes(filename):
    r"""
    Reads one dimensional binary spike file and returns a TD Events.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 40 bit number.
        * First 16 bits (bits 39-24) represent the neuron_id.
        * Bit 23 represents the sign of spike Events: 0=>OFF Events, 1=>ON Events.
        * the last 23 bits (bits 22-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> TD = file_io.read_1d_spikes(file_path)
    """
    with open(filename, "rb") as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_events = (input_as_int[0::5] << 8) | input_as_int[1::5]
    p_events = input_as_int[2::5] >> 7
    t_events = (
        (input_as_int[2::5] << 16) | (input_as_int[3::5] << 8) | (input_as_int[4::5])
    ) & 0x7FFFFF
    return Events(
        x_events, None, p_events, t_events / 1000
    )  # convert spike times to ms


def encode_1d_spikes(filename, TD):
    r"""
    Writes one dimensional binary spike file from a TD Events.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 40 bit number.
        * First 16 bits (bits 39-24) represent the neuron_id.
        * Bit 23 represents the sign of spike Events: 0=>OFF Events, 1=>ON Events.
        * the last 23 bits (bits 22-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.
        * ``TD`` (an ``file_io.Events``): TD Events.

    Usage:

    >>> file_io.write1Dspikes(file_path, TD)
    """
    assert TD.dim != 1, f"Expected TD dimension to be 1. It was: {TD.dim}"
    x_events = np.round(TD.x).astype(int)
    p_events = np.round(TD.p).astype(int)
    t_events = np.round(TD.t * 1000).astype(int)  # encode spike time in us
    output_byte_array = bytearray(len(t_events) * 5)
    output_byte_array[0::5] = np.uint8((x_events >> 8) & 0xFF00).tobytes()
    output_byte_array[1::5] = np.uint8((x_events & 0xFF)).tobytes()
    output_byte_array[2::5] = np.uint8(
        ((t_events >> 16) & 0x7F) | (p_events.astype(int) << 7)
    ).tobytes()
    output_byte_array[3::5] = np.uint8((t_events >> 8) & 0xFF).tobytes()
    output_byte_array[4::5] = np.uint8(t_events & 0xFF).tobytes()
    with open(filename, "wb") as output_file:
        output_file.write(output_byte_array)


#########################################################
# 2D reading and encoding
#########################################################
def read_2d_spikes(filename):
    r"""
    Reads two dimensional binary spike file and returns a TD Events.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike Events: 0=>OFF Events, 1=>ON Events.
        * The last 23 bits (bits 22-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> TD = file_io.read_2d_spikes(file_path)
    """
    with open(filename, "rb") as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_events = input_as_int[0::5]
    y_events = input_as_int[1::5]
    p_events = input_as_int[2::5] >> 7
    t_events = (
        (input_as_int[2::5] << 16) | (input_as_int[3::5] << 8) | (input_as_int[4::5])
    ) & 0x7FFFFF
    return Events(
        x_events, y_events, p_events, t_events / 1000
    )  # convert spike times to ms


def encode_2d_spikes(filename, TD):
    r"""
    Writes two dimensional binary spike file from a TD Events.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike Events: 0=>OFF Events, 1=>ON Events.
        * The last 23 bits (bits 22-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.
        * ``TD`` (an ``file_io.Events``): TD Events.

    Usage:

    >>> file_io.write2Dspikes(file_path, TD)
    """
    assert TD.dim != 2, f"Expected TD dimension to be 2. It was: {TD.dim}"
    x_events = np.round(TD.x).astype(int)
    y_events = np.round(TD.y).astype(int)
    p_events = np.round(TD.p).astype(int)
    t_events = np.round(TD.t * 1000).astype(int)  # encode spike time in us
    output_byte_array = bytearray(len(t_events) * 5)
    output_byte_array[0::5] = np.uint8(x_events).tobytes()
    output_byte_array[1::5] = np.uint8(y_events).tobytes()
    output_byte_array[2::5] = np.uint8(
        ((t_events >> 16) & 0x7F) | (p_events.astype(int) << 7)
    ).tobytes()
    output_byte_array[3::5] = np.uint8((t_events >> 8) & 0xFF).tobytes()
    output_byte_array[4::5] = np.uint8(t_events & 0xFF).tobytes()
    with open(filename, "wb") as output_file:
        output_file.write(output_byte_array)


#########################################################
# 3D reading and encoding
#########################################################
def read_3d_spikes(filename):
    r"""
    Reads binary spike file for spike Events in height, width and channel dimension and returns a TD Events.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 56 bit number.
        * First 12 bits (bits 56-44) represent the xID of the neuron.
        * Next 12 bits (bits 43-32) represent the yID of the neuron.
        * Next 8 bits (bits 31-24) represents the channel ID of the neuron.
        * The last 24 bits (bits 23-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.

    Usage:

    >>> TD = file_io.read_3d_spikes(file_path)
    """
    with open(filename, "rb") as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    x_events = (input_as_int[0::7] << 4) | (input_as_int[1::7] >> 4)
    y_events = (input_as_int[2::7]) | ((input_as_int[1::7] & 0x0F) << 8)
    p_events = input_as_int[3::7]
    t_events = (
        (input_as_int[4::7] << 16) | (input_as_int[5::7] << 8) | (input_as_int[6::7])
    )
    return Events(
        x_events, y_events, p_events, t_events / 1000
    )  # convert spike times to ms


def encode_3d_spikes(filename, TD):
    r"""
    Writes binary spike file for TD Events in height, width and channel dimension.
    
    The binary file is encoded as follows:
        * Each spike Events is represented by a 56 bit number.
        * First 12 bits (bits 56-44) represent the xID of the neuron.
        * Next 12 bits (bits 43-32) represent the yID of the neuron.
        * Next 8 bits (bits 31-24) represents the channel ID of the neuron.
        * The last 24 bits (bits 23-0) represent the spike Events timestamp in microseconds.

    Arguments:
        * ``filename`` (``string``): path to the binary file.
        * ``TD`` (an ``file_io.Events``): TD Events.

    Usage:

    >>> file_io.write3Dspikes(file_path, TD)
    """
    assert TD.dim != 2, f"Expected TD dimension to be 2. It was: {TD.dim}"
    x_events = np.round(TD.x).astype(int)
    y_events = np.round(TD.y).astype(int)
    p_events = np.round(TD.p).astype(int)
    t_events = np.round(TD.t * 1000).astype(int)  # encode spike time in us
    output_byte_array = bytearray(len(t_events) * 7)
    output_byte_array[0::7] = np.uint8(x_events >> 4).tobytes()
    output_byte_array[1::7] = np.uint8(
        ((x_events << 4) & 0xFF) | (y_events >> 8) & 0xFF00
    ).tobytes()
    output_byte_array[2::7] = np.uint8(y_events & 0xFF).tobytes()
    output_byte_array[3::7] = np.uint8(p_events).tobytes()
    output_byte_array[4::7] = np.uint8((t_events >> 16) & 0xFF).tobytes()
    output_byte_array[5::7] = np.uint8((t_events >> 8) & 0xFF).tobytes()
    output_byte_array[6::7] = np.uint8(t_events & 0xFF).tobytes()
    with open(filename, "wb") as output_file:
        output_file.write(output_byte_array)


#########################################################
# 1D reading and encoding of number of spikes
#########################################################
def read_1d_num_spikes(filename):
    r"""
    Reads a tuple specifying neuron, start of spike region, end of spike region and number of spikes from binary spike file.
    
    The binary file is encoded as follows:
        * Number of spikes data is represented by an 80 bit number.
        * First 16 bits (bits 79-64) represent the neuron_id.
        * Next 24 bits (bits 63-40) represents the start time in microseconds.
        * Next 24 bits (bits 39-16) represents the end time in microseconds.
        * Last 16 bits (bits 15-0) represents the number of spikes.
    
    Arguments:
        * ``filename`` (``string``): path to the binary file

    Usage:

    >>> n_id, t_start, t_end, n_spikes = file_io.read_1d_num_spikes(file_path)
    ``t_start`` and ``t_end`` are returned in milliseconds
    """
    with open(filename, "rb") as input_file:
        input_byte_array = input_file.read()
    input_as_int = np.asarray([x for x in input_byte_array])
    neuron_id = (input_as_int[0::10] << 8) | input_as_int[1::10]
    t_startart = (
        (input_as_int[2::10] << 16) | (input_as_int[3::10] << 8) | (input_as_int[4::10])
    )
    t_end = (
        (input_as_int[5::10] << 16) | (input_as_int[6::10] << 8) | (input_as_int[7::10])
    )
    n_spikesikes = (input_as_int[8::10] << 8) | input_as_int[9::10]
    return (
        neuron_id,
        t_startart / 1000,
        t_end / 1000,
        n_spikesikes,
    )  # convert spike times to ms


def encode_1d_num_spikes(filename, n_id, t_start, t_end, n_spikes):
    r"""
    Writes binary spike file given a tuple specifying neuron, start of spike region, end of spike region and number of spikes.
    
    The binary file is encoded as follows:
        * Number of spikes data is represented by an 80 bit number
        * First 16 bits (bits 79-64) represent the neuron_id
        * Next 24 bits (bits 63-40) represents the start time in microseconds
        * Next 24 bits (bits 39-16) represents the end time in microseconds
        * Last 16 bits (bits 15-0) represents the number of spikes
    
    Arguments:
        * ``filename`` (``string``): path to the binary file
        * ``n_id`` (``numpy array``): neuron ID
        * ``t_start`` (``numpy array``): region start time (in milliseconds)
        * ``t_end`` (``numpy array``): region end time (in milliseconds)
        * ``n_spikes`` (``numpy array``): number of spikes in the region

    Usage:

    >>> file_io.encode_1d_num_spikes(file_path, n_id, t_start, t_end, n_spikes)
    """
    neuron_id = np.round(n_id).astype(int)
    t_startart = np.round(t_start * 1000).astype(int)  # encode spike time in us
    t_end = np.round(t_end * 1000).astype(int)  # encode spike time in us
    n_spikesikes = np.round(n_spikes).astype(int)
    output_byte_array = bytearray(len(neuron_id) * 10)
    output_byte_array[0::10] = np.uint8(neuron_id >> 8).tobytes()
    output_byte_array[1::10] = np.uint8(neuron_id).tobytes()
    output_byte_array[2::10] = np.uint8(t_startart >> 16).tobytes()
    output_byte_array[3::10] = np.uint8(t_startart >> 8).tobytes()
    output_byte_array[4::10] = np.uint8(t_startart).tobytes()
    output_byte_array[5::10] = np.uint8(t_end >> 16).tobytes()
    output_byte_array[6::10] = np.uint8(t_end >> 8).tobytes()
    output_byte_array[7::10] = np.uint8(t_end).tobytes()
    output_byte_array[8::10] = np.uint8(n_spikesikes >> 8).tobytes()
    output_byte_array[9::10] = np.uint8(n_spikesikes).tobytes()
    with open(filename, "wb") as output_file:
        output_file.write(output_byte_array)


#########################################################
# Visualize spikes
#########################################################
def _show_td_1d(TD, frame_rate=24, pre_compute_frames=True, repeat=False):
    assert TD.dim != 1, f"Expected TD dimension to be 1. It was: {TD.dim}"
    fig = plt.figure()
    interval = 1e3 / frame_rate  # in ms
    x_dim = TD.x.max() + 1
    tMax = TD.t.max()
    tMin = TD.t.min()
    pMax = TD.p.max() + 1
    min_frame = int(np.floor(tMin / interval))
    max_frame = int(np.ceil(tMax / interval)) + 1

    # ignore pre_compute_frames

    def animate(i):
        fig.clear()
        t_end = (i + min_frame + 1) * interval
        ind = TD.t < t_end
        # plot raster
        plt.plot(TD.t[ind], TD.x[ind], ".")
        # plt.plot(TD.t[ind], TD.x[ind], '.', c=cm.hot(TD.p[ind]))
        # plot raster scan line
        plt.plot([t_end + interval, t_end + interval], [0, x_dim])
        plt.axis((tMin - 0.1 * tMax, 1.1 * tMax, -0.1 * x_dim, 1.1 * x_dim))
        plt.draw()

    anim = animation.FuncAnimation(
        fig, animate, frames=max_frame, interval=42, repeat=repeat
    )  # 42 means playback at 23.809 fps

    return anim


def _show_td_2d(TD, frame_rate=24, pre_compute_frames=True, repeat=False):
    assert TD.dim != 2, f"Expected TD dimension to be 2. It was: {TD.dim}"
    fig = plt.figure()
    interval = 1e3 / frame_rate  # in ms
    x_dim = TD.x.max() + 1
    y_dim = TD.y.max() + 1

    if pre_compute_frames is True:
        min_frame = int(np.floor(TD.t.min() / interval))
        max_frame = int(np.ceil(TD.t.max() / interval))
        image = plt.imshow(np.zeros((y_dim, x_dim, 3)))
        frames = np.zeros((max_frame - min_frame, y_dim, x_dim, 3))

        # precompute frames
        for i in range(len(frames)):
            t_startart = (i + min_frame) * interval
            t_end = (i + min_frame + 1) * interval
            time_mask = (TD.t >= t_startart) & (TD.t < t_end)
            r_ind = time_mask & (TD.p == 1)
            g_ind = time_mask & (TD.p == 2)
            b_ind = time_mask & (TD.p == 0)
            frames[i, TD.y[r_ind], TD.x[r_ind], 0] = 1
            frames[i, TD.y[g_ind], TD.x[g_ind], 1] = 1
            frames[i, TD.y[b_ind], TD.x[b_ind], 2] = 1

        def animate(frame):
            image.set_data(frame)
            return image

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=interval, repeat=repeat
        )

    else:
        min_frame = int(np.floor(TD.t.min() / interval))

        def animate(i):
            t_startart = (i + min_frame) * interval
            t_end = (i + min_frame + 1) * interval
            frame = np.zeros((y_dim, x_dim, 3))
            time_mask = (TD.t >= t_startart) & (TD.t < t_end)
            r_ind = time_mask & (TD.p == 1)
            g_ind = time_mask & (TD.p == 2)
            b_ind = time_mask & (TD.p == 0)
            frame[TD.y[r_ind], TD.x[r_ind], 0] = 1
            frame[TD.y[g_ind], TD.x[g_ind], 1] = 1
            frame[TD.y[b_ind], TD.x[b_ind], 2] = 1
            plot = plt.imshow(frame)
            return plot

        anim = animation.FuncAnimation(
            fig, animate, interval=interval, repeat=repeat
        )  # 42 means playback at 23.809 fps

    return anim

    # # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # # installed.  The extra_args ensure that the x264 codec is used, so that
    # # the video can be embedded in html5.  You may need to adjust this for
    # # your system: for more information, see
    # # http://matplotlib.sourceforge.net/api/animation_api.html
    # if saveAnimation: anim.save('show_td_animation.mp4', fps=30)


def show_td(TD, frame_rate=24, pre_compute_frames=True, repeat=False):
    """
    Visualizes TD Events.

    Arguments:
        * ``TD``: spike Events to visualize.
        * ``frame_rate``: framerate of visualization.
        * ``pre_compute_frames``: flag to enable precomputation of frames for faster visualization. Default is ``True``.
        * ``repeat``: flag to enable repeat of animation. Default is ``False``.

    Usage:

    >>> show_td(TD)
    """
    if TD.dim == 1:
        anim = _show_td_1d(
            TD,
            frame_rate=frame_rate,
            pre_compute_frames=pre_compute_frames,
            repeat=repeat,
        )
    else:
        anim = _show_td_2d(
            TD,
            frame_rate=frame_rate,
            pre_compute_frames=pre_compute_frames,
            repeat=repeat,
        )
    return anim


# def spike_mat2TD(spike_mat, sampling_time=1):		# Sampling time in ms
# 	addressEvents = np.argwhere(spike_mat > 0)
# 	# print(addressEvents.shape)
# 	return Events(addressEvents[:,2], addressEvents[:,1], addressEvents[:,0], addressEvents[:,3] * sampling_time)
