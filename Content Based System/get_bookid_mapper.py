import csv

def get_mapper(filename):
    mapper = {}
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            mapper[line[1]] = line[0]
    return mapper