FIXMATCH_RANDAUGMENT_OPS_LIST = [
    'identity',
    'autocontrast',
    'brightness',
    'color',
    'contrast',
    'equalize',
    'posterize',
    'rotate',
    'sharpness',
    'shear_x',
    'shear_y',
    'solarize',
    'translate_x',
    'translate_y'
]


UDA_RANDAUGMENT_OPS_LIST = [
    'invert',
    'autocontrast',
    'brightness',
    'color',
    'contrast',
    'cutout',
    'equalize',
    'posterize',
    'rotate',
    'sharpness',
    'shear_x',
    'shear_y',
    'solarize',
    'translate_x',
    'translate_y'
]


RANDAUGMENT_MAX_LEVELS = {
    'autocontrast': None,
    'brightness': 1.8,
    'color': 1.8,
    'contrast': 1.8,
    'cutout': 20,
    'equalize': None,
    'identity': None,
    'invert': None,
    'posterize': 4,
    'rotate': 30,
    'sharpness': 1.8,
    'shear_x': 0.3,
    'shear_y':0.3,
    'solarize': 256,
    'translate_x': 10,
    'translate_y': 10
}
