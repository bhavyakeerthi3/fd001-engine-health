"""Execute the generated walkthrough using this exact Python interpreter."""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
path = root/"notebooks"/"01_fd001_walkthrough.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, timeout=120, km=manager, resources={"metadata":{"path":str(root)}})
client.execute()
nbformat.write(notebook,path)
print(f"Executed {len([c for c in notebook.cells if c.cell_type == 'code'])} notebook cells")
