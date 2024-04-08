from logging import getLogger

from ssl_models.mae_module import MAEModule

logger = getLogger(__name__)


def test_mae_module():
    foret_module = MAEModule()
    foret_module.setup(stage="fit")
