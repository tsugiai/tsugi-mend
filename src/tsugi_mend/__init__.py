"""tsugiai-mend-sdk public API.

Public symbols:
    MendConfig       configuration dataclass
    mend_init        runtime initialization
    mend_shutdown    runtime teardown

Patent-independent by deliberate construction. See LICENSE preamble.

Note: MendConfig is torch-free and imports without torch installed (so
tests / configuration / docs tooling and the unified tsugi meta-package
work in lighter environments). mend_init and mend_shutdown lazy-import
torch via tsugi_mend.runtime, which imports torch.nn at module level.
This mirrors the tsugi_kpool facade pattern.
"""
from tsugi_mend.config import MendConfig

__version__ = "0.1.5"


def mend_init(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_mend.runtime import mend_init as _impl
    return _impl(*args, **kwargs)


def mend_shutdown(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_mend.runtime import mend_shutdown as _impl
    return _impl(*args, **kwargs)


__all__ = ["MendConfig", "mend_init", "mend_shutdown", "__version__"]
