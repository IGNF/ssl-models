from ssl_models.mae import MAE


def test_mae():
    mae = MAE()
    mae.setup(stage="fit")
