#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>

using namespace std;

class Perceptron
{
private:
  float bias;
  vector<float> weights;
  float learning_rate;
  static mt19937 gen;

public:
  int predict(vector<float> inputs){
    float z = bias; 
    for (int i = 0; i < weights.size(); i++)
    {
      z += weights[i] * inputs[i];
    }
    return activation(z);
  }
  int activation(float x)
  {
    return (x > 0) ? 1 : 0;
  }

  Perceptron(int num_inputs) : learning_rate(0.05)
  {
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    weights.resize(num_inputs);
    for (auto &w : weights)
    {
      w = dist(gen);
    }
    bias = dist(gen);
  }

  void train(int max_epochs, const vector<vector<float>> &x, const vector<float> &y)
  {
    for (int epoch = 0; epoch < max_epochs; epoch++)
    {
      int total_error = 0;
      for (int i = 0; i < x.size(); i++)
      {
        int pred = predict(x[i]);
        float err = y[i] - pred;
        total_error += abs(err);
        for (int k = 0; k < weights.size(); k++)
        {
          weights[k] += learning_rate * x[i][k] * err;
        }
        bias += learning_rate * err;
      }
      if (total_error == 0)
      {
        cout << "Detenido en el epoch" << epoch + 1 << endl;
        break;
      }
    }
  }

  void print_weights()
  {
    cout << "Pesos: ";
    for (const auto &w : weights)
    {
      cout << w << "\t";
    }
    cout << endl
         << "Bias: " << bias << endl;
  }
  void save_model(const string &filename)
  {
    ofstream file(filename);
    if (file.is_open())
    {
      for (const auto &w : weights)
      {
        file << w << ",";
      }
      file << bias << "\n";
      file.close();
      cout << "Modelo guardado en: " << filename << endl;
    }
    else
    {
      cerr << "Error al abrir el archivo para escritura." << endl;
    }
  }

  void load_model(const string &filename)
  {
    ifstream file(filename);
    if (file.is_open())
    {
      string line;
      getline(file, line);
      stringstream ss(line);
      string value;
      vector<float> loaded_weights;

      while (getline(ss, value, ','))
      {
        if (!value.empty())
        {
          loaded_weights.push_back(stof(value));
        }
      }

      if (!loaded_weights.empty())
      {
        bias = loaded_weights.back();
        loaded_weights.pop_back();
        weights = loaded_weights;
      }
      file.close();
      cout << "Modelo cargado desde: " << filename << endl;
    }
    else
    {
      cerr << "Error al abrir el archivo para lectura." << endl;
    }
  }
};

mt19937 Perceptron::gen(98);

int main()
{
  Perceptron p_and(2);
  Perceptron p_or(2);

  vector<vector<float>> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<float> y_and = {0, 0, 0, 1};
  vector<float> y_or = {0, 1, 1, 1};

  cout << "\tAND" << endl;
  p_and.train(50, x, y_and);
  p_and.print_weights();
  for (int i = 0; i < x.size(); i++)
  {
    cout << "Predict de: " << x[i][0] << ", " << x[i][1]
         << " = " << p_and.predict(x[i]) << "\t"
         << "valor real es: " << y_and[i] << endl;
  }
  p_and.save_model("result/cpp_and_model.txt");


  cout << "\n\n\tOR\t" << endl;
  p_or.train(50, x, y_or);
  p_or.print_weights();
  for (int i = 0; i < x.size(); i++)
  {
    cout << "Predict de: " << x[i][0] << ", " << x[i][1]
         << " = " << p_or.predict(x[i]) << "\t"
         << "valor real es: " << y_or[i] << endl;
  }
  p_or.save_model("result/cpp_or_model.txt");

  return 0;
}