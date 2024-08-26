import numpy as np

if __name__ == '__main__':
    matrix = np.zeros((300, 20000))
    index = 0
    with open('dexter_train_labels.csv', 'r') as file:
        for line in file:
            matrix[index][0] = int(line.strip())
            index += 1
    index = 0
    with open('dexter_train.csv', 'r') as file:
        for line in file:
            entries = line.split(" ")
            for entry in entries:
                data = entry.split(":")
                if(not data[0].isdigit()):
                    print("-------------------")
                    print(entry)
                    print("-------------------")
                    continue
                column_index = int(data[0])
                if 0 <= column_index < 20000:
                    matrix[index][column_index] = int(data[1])
                else:
                    print(column_index)
            index += 1
    print(matrix)
    with open('dexter.csv', 'w+') as file:
        header = 'class,' + ','.join([f'c_{n}' for n in range(1, 20000)])
        file.write(header + '\n')
        for entry in matrix:
            row = ""
            for i in range(0, 20000):
                row = row + str(int(entry[i])) + ','
            row = row[:-1] + '\n'
            print(row)
            file.write(row)
