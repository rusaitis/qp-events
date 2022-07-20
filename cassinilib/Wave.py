

class Wave(object):
    ''' Wave properties for artifical signals '''
    def __init__(self,
                 period=1.,
                 amplitude=1.,
                 phase=0.,
                 type='sine',
                 decay_width=0.,
                 quality=0.,
                 shift=0,
                 cutoff=None,
                 ):
      self.period = period
      self.amplitude = amplitude
      self.phase = phase
      self.type = type
      self.decay_width = decay_width
      self.quality = quality
      self.shift = shift
      self.cutoff = cutoff


