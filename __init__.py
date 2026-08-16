"""ComfyUI custom node package: MiniMax Music 3 (generate + continuation/extend)."""

# Imported for its side effect: registering /mm3/player, /mm3/streams and /mm3/chunk on
# PromptServer.instance.routes. main.py:521 runs init_extra_nodes before main.py:535 calls add_routes(), so
# import-time registration is picked up (and mirrored under /api). nodes.py imports it too; this is explicit.
from . import mm3_stream  # noqa: F401
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serve web/ as this pack's frontend extension directory. The directory must EXIST at startup - the
# registration is guarded on os.path.isdir and a missing folder is silently ignored.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
