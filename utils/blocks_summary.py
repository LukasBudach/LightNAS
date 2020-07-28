import os


def main():
    training_dir = os.path.dirname(os.path.realpath('../__file__')) + '/trainings/'
    for folder in os.listdir(training_dir):
        graph_dir = training_dir + folder + '/logs/architectures/'
        summary_table = []

        if (not os.path.isdir(graph_dir)) or ('meliusnet' not in str(folder)):
            continue
	
        for txt_filepath in sorted([x for x in os.listdir(graph_dir) if x.endswith('txt')]):

            full_path = graph_dir + txt_filepath
            f = open(full_path, "r")
            architecture = f.read()

            num_used_improvement_blocks = architecture.count('ImprovementBlockEnas(\n  (body)')
            num_used_dense_blocks = 0

            for i in range(len(architecture)):
                if architecture[i:i + 14] == 'DenseBlockEnas':
                    type_of_convolution = architecture[i + 314:i + 352]
                    if type_of_convolution.find('(1, 1)') == -1:
                        num_used_dense_blocks += 1

            row = (txt_filepath.split('.txt')[0], num_used_dense_blocks, num_used_improvement_blocks)
            summary_table.append(row)
            print(full_path)
            print(summary_table)

        f = open(graph_dir + 'block_summary.csv', 'w')
        f.write('epochs,used_dense,used_improvement\n')
        for t in summary_table:
            line = ','.join(str(x) for x in t)
            f.write(line + '\n')
        f.close()


if __name__ == "__main__":
    main()
