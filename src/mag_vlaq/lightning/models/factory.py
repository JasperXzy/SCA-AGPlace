from mag_vlaq.models.modeldb import ModelDB
from mag_vlaq.models.modelq import ModelQ


def build_models(cfg):
    db_model = ModelDB(mode="db", dim=cfg.features_dim, args=cfg)
    query_model = ModelQ(args=cfg)
    return db_model, query_model
