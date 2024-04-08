from logging import getLogger

from odeon.core.app_utils import Stages

from ssl_models.mae_module import MAEModule

logger = getLogger(__name__)


def test_mae_module():
    foret_module = MAEModule()
    foret_module.setup(stage=str(Stages))
