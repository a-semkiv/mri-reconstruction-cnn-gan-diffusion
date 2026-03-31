from fastmri.pl_modules import UnetModule


def build_generator():

    return UnetModule(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.0, 
    )
