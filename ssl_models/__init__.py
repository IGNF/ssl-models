from odeon.core import OdnPlugin, PluginMaturity  # type: ignore
from odeon.models import MODEL_REGISTRY  # type: ignore

from .ssl_module import SSLModule

ssl_plugin = OdnPlugin(
    name="input",
    plugin_maturity=str(PluginMaturity.DEVELOPMENT.value),
    version="0.1",
    elements={
        "ssl_module": {
            "class": SSLModule,
            "registry": MODEL_REGISTRY,
            "aliases": ["sslm"],
        }
    },
)
