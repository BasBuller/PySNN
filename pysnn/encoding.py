import torch


############################
# Poisson
############################
def poisson_encoding(intensities, duration, dt):
    r"""Encode a set of spiking intensities to spike trains using a Poisson distribution.
    
    Adapted from:
        https://github.com/Hananel-Hazan/bindsnet/blob/master/bindsnet/encoding/encodings.py

    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """

    assert (intensities >= 0).all(), "Inputs must be non-negative."
    assert intensities.dtype == torch.float, "Intensities must be of type Float."

    # Get shape and size of data.
    shape, size = intensities.shape, intensities.numel()
    intensities = intensities.view(-1)
    time = int(duration / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    non_zero = intensities != 0
    rate[non_zero] = 1 / intensities[non_zero] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, intensities != 0] += (intervals[:, intensities != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).bool()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]
    spikes = spikes.permute(1, 0)

    return spikes.view(*shape, time)


class PoissonEncoder:
    r"""Encode a set of spiking intensities to spike trains using a Poisson distribution.
    
    Adapted from:
        https://github.com/Hananel-Hazan/bindsnet/blob/master/bindsnet/encoding/encodings.py

    Creates a callable PoissonEncoder which encodes as defined in ``bindsnet.encoding.poisson``

    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    """

    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt

    def __call__(self, intensities):
        return poisson_encoding(intensities, self.duration, self.dt)
