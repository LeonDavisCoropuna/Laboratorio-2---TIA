#include <iostream>
#include <vector>
#include <random>
#include "models/perceptron.h"
#include "models/perceptron_torch.h"
using namespace std;

int main() {
    // Datos para la compuerta lógica AND
    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    // vector<int> Y = {0, 0, 0, 1}; // Salidas deseadas para AND
    vector<int> Y = {0, 1, 1, 1}; // Salidas deseadas para OR

    Perceptron p(2);
    PerceptronTorch perceptron_torch(2);
    p.print_weights();

    p.train(X, Y, 100, true); // 100 épocas de entrenamiento

    p.print_weights();

    cout << "Pruebas:\n";
    for (const auto& x : X)
        cout << x[0] << " " << x[1] << " => " << p.predict(x) << endl;

    return 0;
}
