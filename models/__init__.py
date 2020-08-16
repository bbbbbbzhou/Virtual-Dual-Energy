from models import model_DE


def create_model(opts):
    if opts.model_type == 'model_bone':
        model = model_DE.GANModel(opts)

    elif opts.model_type == 'model_softtissue':
        model = model_DE.GANModel(opts)

    else:
        raise NotImplementedError

    return model
