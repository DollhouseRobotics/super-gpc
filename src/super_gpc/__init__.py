"""Stage 1 tracker training for the Dollhouse G1 on MJLab."""

# Import MJLab first: it loads ``mjlab.tasks`` plugins (including this package's
# tasks) on import, which must not run while one of our modules is half-built.
import mjlab  # noqa: F401
