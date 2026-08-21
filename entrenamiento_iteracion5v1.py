"""Wrapper de compatibilidad; el pipeline mantenido vive en ``src/campis``."""

from run import main


if __name__ == "__main__":
    raise SystemExit(main())
