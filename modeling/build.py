from modeling.arch import IFRNet, CIFR_Encoder, Discriminator, PatchDiscriminator, MLP, PatchSampleF


def build_model(args):
    if args.MODEL.NAME.lower() == "ifrnet":
        net = IFRNet(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, destyler_n_channels=args.MODEL.IFR.DESTYLER_CHANNELS)
        mlp = MLP(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, out_features=args.MODEL.NUM_CLASS)
    elif args.MODEL.NAME.lower() == "cifr":
        net = CIFR_Encoder(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, destyler_n_channels=args.MODEL.IFR.DESTYLER_CHANNELS)
        mlp = None
    elif args.MODEL.NAME.lower() == "ifr-no-aux":
        net = IFRNet(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, destyler_n_channels=args.MODEL.IFR.DESTYLER_CHANNELS)
        mlp = None
    else:
        raise NotImplementedError
    return net, mlp


def build_discriminators(args):
    return Discriminator(base_n_channels=args.MODEL.D.NUM_CHANNELS), PatchDiscriminator(base_n_channels=args.MODEL.D.NUM_CHANNELS)


def build_patch_sampler(args):
    return PatchSampleF(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, style_or_content="content", use_mlp=args.MODEL.PATCH.USE_MLP, nc=args.MODEL.PATCH.NUM_CHANNELS), \
           PatchSampleF(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, style_or_content="style", use_mlp=args.MODEL.PATCH.USE_MLP, nc=args.MODEL.PATCH.NUM_CHANNELS)
