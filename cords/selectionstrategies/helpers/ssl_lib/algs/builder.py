from .ict import ICT
from .consistency import ConsistencyRegularization
from .pseudo_label import PseudoLabel
from .vat import VAT


def gen_ssl_alg(name, cfg):
    if name == "ict": # mixed target <-> mixed input
        return ICT(
            cfg.consistency,
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax,
            cfg.alpha
        )
    elif name == "cr": # base augment <-> another augment
        return ConsistencyRegularization(
            cfg.consistency,
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax
        )
    elif name == "pl": # hard label <-> strong augment
        return PseudoLabel(
            cfg.consistency,
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax
        )
    elif name == "vat": # base augment <-> adversarial
        from ..consistency import builder
        return VAT(
            cfg.consistency,
            cfg.threshold,
            cfg.sharpen,
            cfg.temp_softmax,
            builder.gen_consistency(cfg.consistency, cfg),
            cfg.eps,
            cfg.xi,
            cfg.vat_iter
        )
    else:
        raise NotImplementedError
