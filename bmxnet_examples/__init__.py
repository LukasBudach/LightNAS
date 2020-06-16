import sys
import pathlib
#making imports from inside the bmxnet_examples work as this would be the top directory
bmxnet_examples_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(bmxnet_examples_path))