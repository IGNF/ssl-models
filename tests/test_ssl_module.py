from ssl_models.ssl_module import SSLModule


def test_mae():
    ssl_module = SSLModule()
    ssl_module.setup(stage="fit")
