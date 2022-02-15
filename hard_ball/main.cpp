#include "vec.hpp"
#include "input_processing.hpp"

int main() {
    Vec<double, 3> v3(0.);
    for (int i = 0; i < 1000000; i++) {
        Vec<double, 3> v1 = {1, 2, 3};
        Vec<double, 3> v2 = {4, 5, 6};
        v3 += v1 / v2;
    }
    std::cout << v3 << std::endl;
    return 0;
}
