"""Compatibilidad con el nombre usado antes de modularizar el proyecto.

Ejecutado sin argumentos muestra la ayuda del nuevo CLI. También acepta los
mismos subcomandos que ``python run.py``; por ejemplo::

    python entrenamiento_iteracion5v1_vscode.py train --config configs/default.toml
"""

from run import main


if __name__ == "__main__":
    raise SystemExit(main())
