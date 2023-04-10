import typing as tp

from torch.optim import Adam
from modelling.autoencoders import TimeDomainAE, FrequencyDomainAE


TModels = tp.Dict[str, tp.Union[TimeDomainAE, FrequencyDomainAE]]
TOptimizers = tp.Dict[str, Adam]


def get_models(config: tp.Dict[str, tp.Any]) -> TModels:
    time_domain_model: TimeDomainAE = TimeDomainAE(
        config["modelling"]["window_size"],
        config["modelling"]["num_time_invariant_features"],
        config["modelling"]["num_instantaneous_features"],
    ).to(config["training"]["device"])

    frequency_domain_model: FrequencyDomainAE = FrequencyDomainAE(
        config["modelling"]["dft_shape"] // 2 + 1,
        config["modelling"]["num_time_invariant_features"],
        config["modelling"]["num_instantaneous_features"],
    ).to(config["training"]["device"])
    return {"td": time_domain_model, "fd": frequency_domain_model}


def get_optmizers(models: TModels, config: tp.Dict[str, tp.Any]) -> TOptimizers:
    time_domain_optimizer: Adam = Adam(
        models["td"].parameters(),
        lr=config["modelling"]["learning_rate"],
        weight_decay=config["modelling"]["weight_decay"],
    )

    frequency_domain_optimizer: Adam = Adam(
        models["fd"].parameters(),
        lr=config["modelling"]["learning_rate"],
        weight_decay=config["modelling"]["weight_decay"],
    )

    return {"td": time_domain_optimizer, "fd": frequency_domain_optimizer}
