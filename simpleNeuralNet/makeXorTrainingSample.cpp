#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main() {
  // random training sets for XOR
  cout << "topology: 2 4 1" << endl;
  for (int ii = 2000; ii >= 0; --ii) {
    int n1 = (int)(2.0 * rand() / double(RAND_MAX)); // 0 to 1.9999..
    int n2 = (int)(2.0 * rand() / double(RAND_MAX));
    int t = n1 ^ n2; // shoud be 0 or 1; ^ xor
    cout << "in: "<< n1 << ".0 " << n2 << ".0 "<< endl;
    cout << "out: " << t << ".0" << endl;
  }
}
