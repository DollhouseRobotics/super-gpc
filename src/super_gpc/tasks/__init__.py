"""MJLab task plugin: importing this package registers the project tasks."""

from mjlab.utils.lab_api.tasks.importer import import_packages

import_packages(__name__, ["utils", ".mdp", ".rl"])
