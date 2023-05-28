
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'srsc_rsflow_multi':
        from models.model_srsc_rsflow_multi import ModelSRSCRSG as M

    elif model == 'srsc_rsflow_multi_distillv2':
        from models.model_srsc_rsflow_multi_distillv2 import ModelSRSCRSG as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
