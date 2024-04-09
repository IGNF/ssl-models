from odeon.core import OdnPlugin, PluginMaturity
from odeon.models import MODEL_REGISTRY

from .mae_module import MAEModule

ssl_plugin = OdnPlugin(
    name="input",
    plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
    version="0.1",
    elements={
        "ssl_module": {
            "class": MAEModule,
            "registry": MODEL_REGISTRY,
            "aliases": ["sslm"],
        }
    },
)
