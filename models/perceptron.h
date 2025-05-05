#include <iostream>
#include <vector>
#include <random>
using namespace std;

class Perceptron
{
private:
  vector<double> weights;
  double bias;
  double learning_rate;

public:
  Perceptron(int input_size, double lr = 0.1111111)
  {
    weights.resize(input_size, 0.555555);
    bias = 0.555555;
    learning_rate = lr;
  }

  int activation(double z)
  {
    return (z >= 0) ? 1 : 0;
  }

  int predict(const vector<double> &x)
  {
    double z = bias;
    for (size_t i = 0; i < x.size(); ++i)
      z += weights[i] * x[i];
    return activation(z);
  }

  void train(const vector<vector<double>> &X, const vector<int> &y, int epochs, bool show_weights)
  {
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
      double acc_err = 0;
      for (size_t i = 0; i < X.size(); ++i)
      {
        int prediction = predict(X[i]);
        int error = y[i] - prediction;
        acc_err += error;
        // Ajustar pesos y bias
        for (size_t j = 0; j < weights.size(); ++j)
        {
          weights[j] += learning_rate * error * X[i][j];
        }
        bias += learning_rate * error;

        if (show_weights)
        {
          cout << "Epoch " << epoch + 1 << " - Pesos: ";
          for (const auto &w : weights)
          {
            cout << w << " ";
          }
          cout << " | Bias: " << bias << endl;
        }

        if (acc_err <= 0)
          break;        
      }
    }
  }

  void print_weights()
  {
    cout << "Pesos: ";
    for (double w : weights)
      cout << w << " ";
    cout << "\nBias: " << bias << endl;
  }
};
