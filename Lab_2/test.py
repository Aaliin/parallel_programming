import numpy as np


def read_matrix(filename):
    """Осуществляет чтение файла"""
    try:
        with open(filename, 'r') as f:
            matrix = []
            for line in f:
                row = [float(x) for x in line.strip().split()] 
                matrix.append(row)
            return np.array(matrix)
    except Exception as e:
        print(f"Error reading to {filename}: {e}")
        return None


def multiply_matrix(matrix1, matrix2):
    """Осуществляет перемножение матриц"""
    try: 
        if matrix1.shape[1] != matrix2.shape[0]:
            print("Error: The sizes don't match")
            return None
        result_matrix = np.matmul(matrix1, matrix2)
        return result_matrix 
    except Exception as e:
        print(f"Error: Invalid matrix format: {e}")
        return None


def write_matrix(filename, matrix):
    """Осуществляет запись полученной матрицы"""
    try:
        with open(filename, "w") as f:
            for row in matrix:
                f.write(" ".join(map(str, row)) + "\n")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


if __name__ == "__main__":
    input_file1 = "Lab_2\\matrix\\matrix1.txt"
    input_file2 = "Lab_2\\matrix\\matrix2.txt"
    comparison_file = "Lab_2\\matrix\\matrix_res.txt"
    output_file = "Lab_2\\python_res.txt"

    matrix_1 = read_matrix(input_file1)
    matrix_2 = read_matrix(input_file2)
    comparison = read_matrix(comparison_file)

    multi_matrix = multiply_matrix(matrix_1, matrix_2)
    write_matrix(output_file, multi_matrix)

    with open("Lab_2\\comparison_result.txt", "a") as file:
        if np.allclose(comparison, multi_matrix):
            file.write("The size: " + str(comparison.shape[0]) + "\t the result matched")
        else:
            file.write("The result didn't match")
