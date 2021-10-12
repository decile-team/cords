from .ict import ICT
from .consistency import ConsistencyRegularization
from .pseudo_label import PseudoLabel
from .vat import VAT


def gen_ssl_alg(name, cfg):
    if name == "ict": # mixed target <-> mixed input
        return ICT(
            cfg.ssl_args.consis,
            cfg.ssl_args.threshold,
            cfg.ssl_args.sharpen,
            cfg.ssl_args.temp_softmax,
            cfg.ssl_args.alpha
        )
    elif name == "cr": # base augment <-> another augment
        return ConsistencyRegularization(
            cfg.ssl_args.consis,
            cfg.ssl_args.threshold,
            cfg.ssl_args.sharpen,
            cfg.ssl_args.temp_softmax
        )
    elif name == "pl": # hard label <-> strong augment
        return PseudoLabel(
            cfg.ssl_args.consis,
            cfg.ssl_args.threshold,
            cfg.ssl_args.sharpen,
            cfg.ssl_args.temp_softmax
        )
    elif name == "vat": # base augment <-> adversarial
        from ..consistency import builder
        return VAT(
            cfg.ssl_args.consis,
            cfg.ssl_args.threshold,
            cfg.ssl_args.sharpen,
            cfg.ssl_args.temp_softmax,
            builder.gen_consistency(cfg.ssl_args.consis, cfg),
            cfg.ssl_args.eps,
            cfg.ssl_args.xi,
            cfg.ssl_args.vat_iter
        )
    else:
        raise NotImplementedError
