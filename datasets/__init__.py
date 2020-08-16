from . import de_dataset


def get_datasets(opts):
    if opts.dataset == 'DE':
        trainset = de_dataset.DE_Train(opts)
        valset = de_dataset.DE_Test(opts)

    else:
        raise NotImplementedError

    return trainset, valset
