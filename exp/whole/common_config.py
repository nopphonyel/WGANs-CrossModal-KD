class CommonConfig:
    nGPU = 1

    DEV = "cuda"

    # EPOCHS -> Number of training epochs
    # LR -> Learning Rate
    # BS -> Batch Size
    EPOCHS = 5000
    LR = 1e-4
    BS = 16

    # LOAD_GEN -> Specify to load Generator model from checkpoint
    # LOAD_DIS -> Same as above but with Discriminator
    # LOAD_EPOCH -> Specify which checkpoint to load
    LOAD_GEN = False
    LOAD_DIS = False
    LOAD_EPOCH = 0

    # Z_DIM is a size of noise vector being feed to Generator model
    Z_DIM = 100

    # LAMBDA_GP is a hyperparameter in WGANs stuff. ;-;
    LAMBDA_GP = 10

    # Most of GANs, in one epoch, usually train the Discriminator for K
    # times and Generator for 1 time. D_EPCH_STEPS is used for specify K
    # Value. I also create G_EPCH_STEPS for specify Generator training
    # times in one epoch.
    D_EPCH_STEPS = 5
    G_EPCH_STEPS = 1

    # Define the frequency of generated image export preview
    EXPRT_GEN_IMG_FREQ = 20
    # Define the pattern of generation preview image's filename
    EXPRT_GEN_IMG_NAME = "best_fid_epch_{}.png"
    # Define number of generated image to preview the generation
    PREV_GEN_NUM = 10

    # The frequency that FID score being calculated...
    FID_CALC_FREQ = 5
