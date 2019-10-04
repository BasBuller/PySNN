import torch


############################
# Poisson
############################
def poisson_encoding(intensities, duration, dt):
    r"""Encode a set of spiking intensities to spike trains using a Poisson distribution.
    
    Adapted from:
        https://github.com/Hananel-Hazan/bindsnet/blob/master/bindsnet/encoding/encodings.py
    """

    assert (intensities >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = intensities.shape, intensities.numel()
    intensities = intensities.view(-1)
    time = int(duration / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[intensities != 0] = 1 / intensities[intensities != 0] * (1000 / dt)

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

    return spikes.view(time, *shape)


class PoissonEncoder:
    r"""Encode a set of spiking intensities to spike trains using a Poisson distribution.
    
    Adapted from:
        https://github.com/Hananel-Hazan/bindsnet/blob/master/bindsnet/encoding/encodings.py
    """
    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt

    def __call__(self, intensities):
        return poisson_encoding(intensities, self.duration, self.dt)
