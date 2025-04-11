#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>   
#include <ctime>
#include <omp.h>
#include <iomanip> 
#include <sstream>

using namespace std;

const int FLOW_COUNTS[] = { 2, 4, 6, 10, 12 };


void create_matrix_file(const string& filename, const vector<vector<double>>& matrix) {
    ofstream outpu_file(filename);

    if (!outpu_file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        exit(1);
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outpu_file << matrix[i][j] << " ";
        }
        outpu_file << endl;
    }

    outpu_file.close();
}

void create_matrix(string file_name, int row, int col) {
    vector<vector<double>> matrix(row, vector<double>(col, 0.0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i][j] = rand();
        }
    }
    create_matrix_file(file_name, matrix);
}

vector<vector<double>> read_matrix_file(const string& filename, int& rows, int& cols) {
    ifstream input_file(filename);

    if (!input_file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    vector<vector<double>> matrix(rows, vector<double>(cols));

#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            input_file >> matrix[i][j];
        }
    }

    input_file.close();
    return matrix;
}

vector<vector<double>> matrix_multiplication(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2, int flow) {
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int rows2 = matrix2.size();
    int cols2 = matrix2[0].size();

    if (cols1 != rows2) {
        cerr << "Error: The sizes don't match" << endl;
        exit(1);
    }

    vector<vector<double>> result(rows1, vector<double>(cols2, 0.0));

#pragma omp parallel for num_threads(flow)

    for (int i = 0; i < rows1; ++i) {
        for (int k = 0; k < cols1; ++k) {
            double temp = matrix1[i][k];
            for (int j = 0; j < cols2; ++j) {
                result[i][j] += temp * matrix2[k][j];
            }
        }
    }
    return result;
}

void write_matrix_file(const string& filename, const vector<vector<double>>& matrix, double time, int flow) {
    ofstream output_file(filename);
    stringstream time_name;
    time_name << "time_" << flow << ".txt";
    ofstream time_file(time_name.str(), ios::app);

    if (!output_file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        exit(1);
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    time_file << "The size: " << rows << " x " << cols << "\tTotal time: " << time << setprecision(5) << " seconds" << endl;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output_file << matrix[i][j] << " ";
        }
        output_file << endl; 
    }

    time_file.close();
    output_file.close();
}


int main() {
    string matrix_file1, matrix_file2, output_file;
    int rows1, cols1, rows2, cols2;

    cout << "Enter the name of the first matrix file, row_1, col_1: ";
    cin >> matrix_file1 >> rows1 >> cols1;

    cout << "Enter the name of the second matrix file, row_2, col_2: ";
    cin >> matrix_file2 >> rows2 >> cols2;

    cout << "Enter the name of the output file: ";
    cin >> output_file;

    create_matrix(matrix_file1, rows1, cols1);
    create_matrix(matrix_file2, rows2, cols2);

    vector<vector<double>> matrix1 = read_matrix_file(matrix_file1, rows1, cols1);
    vector<vector<double>> matrix2 = read_matrix_file(matrix_file2, rows2, cols2);

    for (int i = 0; i < sizeof(FLOW_COUNTS) / sizeof(FLOW_COUNTS[0]); ++i) {
        int num_threads = FLOW_COUNTS[i];
        omp_set_num_threads(num_threads);

        auto start = chrono::high_resolution_clock::now();
        vector<vector<double>> result_matrix = matrix_multiplication(matrix1, matrix2, num_threads);
        auto end = chrono::high_resolution_clock::now();

        double time = chrono::duration_cast<chrono::microseconds>(end - start).count() * 1e-6;
        write_matrix_file(output_file, result_matrix, time, num_threads);
        cout << "Number of threads " << num_threads << ": " << fixed << setprecision(5) << time << " seconds" << endl;
    }
    return 0;
}