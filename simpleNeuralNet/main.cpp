#include <vector>

using namespace std;

typedef vector<Neuron> Layer;

class Net {
public:
  Net(vecotr<unsigned>& topology);
  void feedForward(const vector<double>& inputVals);
  void backProp(const vector<double>& targetVals);
  void getResults(vector<double>& resultValues) const;
private:
  vector<Layer> m_layers;// m_layers[layerNum][neronNum]

};

int main() {
  vector<unsigned> topology;
  Net myNet(topology);

  vector<double> inputVals;
  myNet.feedForward(inputVals);

  vector<double> targetVals;
  myNet.backProp(targetVals);

  vector<double> resultVals;
  myNet.getResults(resultVals);
}
