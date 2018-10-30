import pathlib
import os

# Get the project directory as the parent of this module location
project_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent

# reports dir
reports_dir = project_dir / "reports"


from . import data, models, features, visualization