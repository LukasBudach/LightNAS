import argparse
import math
from pathlib import Path
import pydot
from tqdm import tqdm


class Color:
    """ Color class for easier use later on (define RGB, mix, get as HEX)
    """

    def __init__(self, r=0, g=0, b=0):
        self._r = Color.clamp(r)
        self._g = Color.clamp(g)
        self._b = Color.clamp(b)

    def r(self):
        return self._r

    def g(self):
        return self._g

    def b(self):
        return self._b

    def set_r(self, val):
        self._r = Color.clamp(val)

    def set_g(self, val):
        self._g = Color.clamp(val)

    def set_b(self, val):
        self._b = Color.clamp(val)

    def as_hex(self):
        return "#{0:02x}{1:02x}{2:02x}".format(self.r(), self.g(), self.b())

    @staticmethod
    def mix(one, other, mix_fac=0.5):
        assert ((type(one) is Color) or (type(other) is Color))
        if (one is None) or (type(one) is not Color):
            return other
        elif (other is None) or (type(other) is not Color):
            return one
        return Color(
            r=Color.clamp(math.sqrt((1 - mix_fac) * math.pow(one.r(), 2) + mix_fac * math.pow(other.r(), 2))),
            g=Color.clamp(math.sqrt((1 - mix_fac) * math.pow(one.g(), 2) + mix_fac * math.pow(other.g(), 2))),
            b=Color.clamp(math.sqrt((1 - mix_fac) * math.pow(one.b(), 2) + mix_fac * math.pow(other.b(), 2)))
        )

    @staticmethod
    def clamp(val):
        return round(max(0, min(val, 255)))


def load_dot_files(file_dir, prefix, max_epoch, min_epoch=0):
    """ Reads all dot files (which seem to actually have no file extension) in the given directory by iterating through
    the given epoch range and appending the epoch to the prefix, using this as file name.

    :param file_dir: directory that contains the dot (graph description) files
    :type file_dir: Path
    :param prefix: everything that is before the epoch number in the dot file names
    :type prefix: str
    :param max_epoch: maximum epoch during training (inclusive)
    :type max_epoch: int
    :param min_epoch: minimum epoch during training, defaults to 0
    type max_epoch: int
    :return: dict mapping filename to pydot.Dot object
    :rtype: dict
    """
    dots = {}
    print('Loading the files...')
    for i in tqdm(range(min_epoch, max_epoch + 1)):
        filename = prefix + str(i) + '.dot'
        dots[filename] = (pydot.graph_from_dot_file(file_dir / filename)[0])
    return dots


def format_and_render(dot_file):
    if type(dot_file) is str:
        dot_file = Path(dot_file)
    graph = pydot.graph_from_dot_file(dot_file)[0]
    node_list = graph.get_node_list()
    for node in node_list:
        format_node(node)
    graph.write_pdf(dot_file.with_suffix('.pdf'))
    graph.write_png(dot_file.with_suffix('.png'))


def format_node(node):
    """ Formats the given node. Currently this is just adding colors based on the label, which represents all of the
    ENAS attributes chosen for any given node.

    :param node: pydot.Node object to be formatted
    :type node: pydot.Node
    :return: nothing, changes are made in the object passed
    """
    attributes = node.get_attributes()
    new_color = None
    if 'label' not in attributes:
        return
    if attributes['label'] == 'node':
        return
    # mix in the color for the skip attribute
    if 'RTrue' in attributes['label']:
        new_color = Color.mix(new_color, Color(r=50, g=255, b=50))
    if 'RFalse' in attributes['label']:
        new_color = Color.mix(new_color, Color(r=255, g=50, b=50))
    # possibly mix in colors for additionally ENAS attributes
    # ...

    # set the new color in the graph
    if new_color:
        node.set_color(new_color.as_hex())


def format_graphs(graphs):
    """ Iterates through the given list of pydot.Dot objects. For each graph object, iterates over its nodes, formatting
    them.

    :param graphs: dictionary of pydot.Dot objects to be formatted
    :type graphs: dict
    :return: nothing, changes are made in the objects themselves
    """
    print('Formatting graphs...')
    for graph in tqdm(graphs.values()):
        node_list = graph.get_node_list()
        for node in node_list:
            format_node(node)


def write_graphs_to_pdf(graphs, out_dir=Path('./')):
    """ Writes the graphs as PDF files into the given directory, using their stored original filenames.

    :param graphs: dictionary of pydot.Dot objects to be written to disk
    :type graphs: dict
    :param out_dir: directory to write to, defaults to the current directory
    :type out_dir: Path
    :return: nothing
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Writing graphs to PDF...')
    for graph_file in tqdm(graphs):
        graphs[graph_file].write_pdf(out_dir / (graph_file + '.pdf'))


def main(arguments):
    graphs = load_dot_files(
        file_dir=Path(arguments.input_dir),
        prefix=arguments.prefix,
        min_epoch=0 if arguments.min_epoch is None else arguments.min_epoch,
        max_epoch=arguments.max_epoch)
    format_graphs(graphs)
    write_graphs_to_pdf(
        graphs=graphs,
        out_dir=Path(arguments.output_dir)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to be used to format all the network graphs that were created '
                                                 'during training with our ENAS.')

    parser.add_argument('-i', '--input-dir', type=str, required=True, help='Path to the directory containing the dot '
                                                                           'files describing the graphs.')
    parser.add_argument('-p', '--prefix', type=str, required=True, help='Part of the log file\'s name before the epoch '
                                                                        'number (e.g. epoch_1 has prefix epoch_)')
    parser.add_argument('--min-epoch', type=int, required=False, help='First epoch that is to be read.')
    parser.add_argument('--max-epoch', type=int, required=True, help='Last epoch to be read, inclusive.')
    parser.add_argument('-o', '--output-dir', type=str, required=False, help='Directory for the Graphs to be written to'
                                                                             ' after formating.')

    args = parser.parse_args()
    main(args)
