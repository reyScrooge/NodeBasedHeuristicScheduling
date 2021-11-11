with open('data2.csv', 'r+') as f:
    # read file
    file_source = f.read()
    replace_string = file_source.replace('|', ',')
    # save output
    f.write(replace_string)

    f.close()

nodes_stats = {}
with open('data2.csv', 'r+') as f:
    line = f.readline()

    line = f.readline()
    while line:
        heu = line.split(',')[0]
        node = line.split(',')[1]
        nodes_stats[node] = {}
        nodes_stats[node][heu] = []
        for i in range(8):
            nodes_stats[node][heu].append(line.split(',')[i+2])

        print(nodes_stats[node][heu])
        line = f.readline()

