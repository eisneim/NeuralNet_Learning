#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection {
  double weight;
  double deltaWeight;
};

class TrainingData {
public:
  TrainingData(const string filename);
  bool isEof(void) { return m_trainingDataFile.eof(); }
  void getTopology(vector<unsigned>& topology);

  // return the number of input values read from the file:
  unsigned getNextInputs(vector<double>& inputVals);
  unsigned getTargetOutputs(vector<double>& targetOutputVals);
private:
  ifstream m_trainingDataFile;
};

TrainingData::TrainingData(const string filename) {
  m_trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned>& topology) {
  string line;
  string label;

  getline(m_trainingDataFile, line);
  stringstream ss(line);
  ss >> label;
  if (this->isEof() || label.compare("topology:") != 0) {
    abort();
  }
  while (!ss.eof()) {
    unsigned n;
    ss >> n;
    topology.push_back(n);
  }
  return ;
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals) {
  inputVals.clear();
  string line;
  getline(m_trainingDataFile, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("in:") == 0) {
    double oneValue;
    while (ss >> oneValue) {
      inputVals.push_back(oneValue);
    }
  }

  return (unsigned)inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals) {
  targetOutputVals.clear();
  string line;
  getline(m_trainingDataFile, line);
  stringstream ss(line);

  string label;
  ss >> label;
  if (label.compare("out:") == 0) {
    double oneValue;
    while (ss >> oneValue) {
      targetOutputVals.push_back(oneValue);
    }
  }
  return (unsigned)targetOutputVals.size();
}

// ---------------------------- Neuron ----------------------------------
class Neuron {
public:
  Neuron(unsigned numOutputs, unsigned myIndex);
  void feedForward(const Layer& prevLayer);
  void setOutputVal(double val) { m_outputVal = val; }
  double getOutputVal(void) const { return m_outputVal; }
  void updateInputWeights(Layer& prevLayer);
  void calcHiddenGradients(const Layer& nextLayer);
  void calcOutputGradients(double targetVal);

private:
  static double eta; // [0.0 .. 1.0] overall net train rate
  static double alpha; // [0.0 .. n] multiplier of last wieght change(momentum)
  // this return 0 - 1 random number
  static double randomWeight(void) { return rand() / double(RAND_MAX); }
  static double transferFunction(double x);
  static double transferFunctionDerivative(double x);
  double sumDow(const Layer& nextLayer);

  double m_outputVal;
  double m_gradient;
  vector<Connection> m_outputWeights;
  unsigned m_myIndex;

};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer& prevLayer) {
  // the weights to be updated are in the connection container
  // in the neurons in the precceding layer
  for (unsigned nn = 0; nn < prevLayer.size(); nn++) {
    Neuron& neuron = prevLayer[nn];
    double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
    double newDeltaWeight =
    // individual input, magnified by the gradient and train rate;
    eta // the over all net learning rate;
    * neuron.getOutputVal()
    * m_gradient
    // also add momentum = a function of the previous delta weight
    + alpha
    * oldDeltaWeight;
    neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
  }
}

double Neuron::sumDow(const Layer& nextLayer) {
  double sum = 0.0;
  // sum our contribution of the errors at the nodes we feed;
  for (unsigned nn = 0; nn < nextLayer.size() - 1; nn++) {
    sum += m_outputWeights[nn].weight * nextLayer[nn].m_gradient;
  }
  return sum;
}

void Neuron::calcHiddenGradients(const Layer& nextLayer) {
  double dow = sumDow(nextLayer);
  m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
  double delta = targetVal - m_outputVal;
  // dy / dx = derivative => dy = dx * derivatigve ?
  m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
  m_myIndex = myIndex;
  for (unsigned ii = 0; ii < numOutputs; ii++) {
    Connection c;
    c.weight = randomWeight();
    m_outputWeights.push_back(c);
  }
}

void Neuron::feedForward(const Layer& prevLayer) {
  double sum = 0.0;

  //sum the previous layer's outputs include bias node
  for (unsigned nn = 0; nn < prevLayer.size(); nn++) {
    Neuron preNu = prevLayer[nn];
    sum += preNu.getOutputVal() *
    preNu.m_outputWeights[m_myIndex].weight;
  }

  m_outputVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x) {
  // tanh - output range: [-1.0, 1.0]
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
  // tanh derivative, approximately
  // d(tanh x)/ dx = 1 - tanh^2 x
  return 1.0 - x * x;
}

// ---------------------------- Net ----------------------------------
class Net {
public:
  Net(vector<unsigned>& topology);
  void feedForward(const vector<double>& inputVals);
  void backProp(const vector<double>& targetVals);
  void getResults(vector<double>& resultVals);
  double getRecentAverageError(void) const { return m_recentAverageError; }
private:
  vector<Layer> m_layers;// m_layers[layerNum][neronNum]
  double m_error;
  double m_recentAverageError;
  double m_recentAverageSmoothFactor;
};

void Net::getResults(vector<double>& resultVals) {
  resultVals.clear();
  Layer& outputLayer = m_layers.back();
  for (unsigned nn = 0; nn < outputLayer.size() - 1; nn++) {
    resultVals.push_back(outputLayer[nn].getOutputVal());
  }
}

Net::Net(vector<unsigned>& topology) {
  unsigned numLayers = (unsigned)topology.size();
  for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
    m_layers.push_back(Layer());
    unsigned numOutputs = numLayers - 1 ? 0 : topology[layerNum + 1];

    // we have made a new layer, now fill it with neurons
    // add add a bias neuron to the layer, so use: '<='
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
      m_layers.back().push_back(Neuron(numOutputs, neuronNum));
      cout << "made a new Neuron" << endl;
    }
  }
}

void Net::feedForward(const vector<double>& inputVals) {
  //input size must match first layer's neuron count
  assert(inputVals.size() == m_layers[0].size() - 1);
  // assign (latch) the input values into the input neurons
  for (unsigned ii = 0; ii < inputVals.size(); ii++) {
    m_layers[0][ii].setOutputVal(inputVals[ii]);
  }
  // forward propagation;  for each layer
  for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
    // for each neuron in this layer
    for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
      Layer& prevLayer = m_layers[layerNum - 1];
      m_layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

void Net::backProp(const vector<double>& targetVals) {
  // calculate overall net error
  // RMS = "root mean square error"
  Layer& outputLayer = m_layers.back();
  m_error = 0.0;

  for (unsigned nn = 0; nn < outputLayer.size() - 1; nn++) {
    double delta = targetVals[nn] - outputLayer[nn].getOutputVal();
    m_error += delta * delta;
  }
  m_error /= outputLayer.size() - 1;// get average error squared
  m_error = sqrt(m_error);

  // implement a recent average measurement
  m_recentAverageError =
  (m_recentAverageError * m_recentAverageSmoothFactor + m_error)
  / (m_recentAverageSmoothFactor + 1.0);

  // calculate output layer gradients
  for (unsigned nn = 0; nn < outputLayer.size() - 1; nn++) {
    outputLayer[nn].calcOutputGradients(targetVals[nn]);
  }

  // calculate gradients on hidden layers
  for (unsigned layerNum = (unsigned)m_layers.size() - 2; layerNum > 0; layerNum--) {
    Layer& hiddenLayer = m_layers[layerNum];
    Layer& nextLayer = m_layers[layerNum + 1];

    for (unsigned nn = 0; nn < hiddenLayer.size(); nn++) {
      hiddenLayer[nn].calcHiddenGradients(nextLayer);
    }
  }

  //for all layers from outputs to first hidden layer,
  //update connection weights
  for (unsigned layerNum = (unsigned)m_layers.size() - 1; layerNum > 0; layerNum--) {
    Layer& layer = m_layers[layerNum];
    Layer& prevLayer = m_layers[layerNum - 1];
    // for each neuron
    for (unsigned nn = 0; nn < layer.size(); nn++) {
      layer[nn].updateInputWeights(prevLayer);
    }
  }
}

void showVectorVals(string label, vector<double> &v)
{
  cout << label << " ";
  for (unsigned i = 0; i < v.size(); ++i) {
    cout << v[i] << " ";
  }

  cout << endl;
}

int main() {
  vector<unsigned> topology;
  TrainingData trainData("simpleNeuralNet/trainingData.txt");
  trainData.getTopology(topology);

  Net myNet(topology);


  vector<double> inputVals, targetVals, resultVals;
  int trainingPass = 0;

  while (!trainData.isEof()) {
    ++trainingPass;
    cout << endl << "Pass " << trainingPass;
    // Get new input data and feed it forward:
    if (trainData.getNextInputs(inputVals) != topology[0])
      break;

    showVectorVals(": Inputs:", inputVals);
    myNet.feedForward(inputVals);

    // collect the net's actual output result
    myNet.getResults(resultVals);
    showVectorVals("outputs:", resultVals);

    // train the net what the output should have been;
    trainData.getTargetOutputs(targetVals);
    showVectorVals("Targets:", targetVals);
    assert(targetVals.size() == topology.back());

    myNet.backProp(targetVals);

    // report how well the training is working, average over samples
    cout << "Net recent average error: "
    << myNet.getRecentAverageError() << endl;
  }

  cout << endl << "Done" << endl;
}
