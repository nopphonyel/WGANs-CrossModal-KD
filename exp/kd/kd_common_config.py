class KDCommonConfig:
    nGPU = 1

    DEV = "cuda"

    # EPOCHS -> Number of training epochs
    # LR -> Learning Rate
    # BS -> Batch Size
    EPOCHS = 5000
    LR = 1e-4
    BS = 16
    BETAS_G = (0.0, 0.9)


    # Number of class of generated images
    NUM_CLASS = 6

    # Size of input noise being feed to GANs
    Z_DIM = 100

    # Size of latent from the feature extractor which being feed
    # to Generator
    LATENT_SIZE = 200

    # Number of generated image channel
    IMG_CHAN = 1
