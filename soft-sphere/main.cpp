#include <limits>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>

using size_t = std::size_t;

struct Vec2 {
    double x;
    double y;

    double &operator[](size_t i);

    double operator[](size_t i) const;

    Vec2 operator+(const Vec2 &v) const;

    Vec2 operator-(const Vec2 &v) const;

    Vec2 operator*(const Vec2 &v) const;

    Vec2 operator/(const Vec2 &v) const;

    Vec2 operator+(double a) const;

    Vec2 operator-(double a) const;

    Vec2 operator*(double a) const;

    Vec2 operator/(double a) const;

    Vec2 operator-() const;

    Vec2 &operator+=(const Vec2 &v);

    Vec2 &operator-=(const Vec2 &v);

    Vec2 &operator*=(double a);

    Vec2 &operator/=(double a);

    Vec2 operator==(const Vec2 &v) const;

    Vec2 operator!=(const Vec2 &v) const;

    Vec2 operator<(const Vec2 &v) const;

    Vec2 operator>(const Vec2 &v) const;

    Vec2 operator<=(const Vec2 &v) const;

    Vec2 operator>=(const Vec2 &v) const;

    double norm() const;

    double norm2() const;

    double dot(const Vec2 &v) const;

    double cross(const Vec2 &v) const;

    static Vec2 unitRand();
};

Vec2 operator+(double a, const Vec2 &v);

Vec2 operator-(double a, const Vec2 &v);

Vec2 operator*(double a, const Vec2 &v);

Vec2 operator/(double a, const Vec2 &v);

std::ostream &operator<<(std::ostream &os, const Vec2 &v);

struct Macrostates {
    double kinetic{};
    double potential{};
    std::vector<double> pressure;
    std::vector<double> temperature;
    struct {
        double mean;
        double variance;
    } pressure_acc{}, temperature_acc{};
};

class SoftSphereSimulation {
public:
    size_t nParticles = 0;
    double time = 0;
    double dt;
    double velMag;
    Vec2 boxSize;
    size_t accumulationSize;
    Macrostates macrostate{};
    Vec2 totalMomentum{};
    Vec2 centerOfMass{};
    Vec2 centerOfMassVelocity{};
    std::vector<Vec2> pos;
    std::vector<Vec2> vel;
    std::vector<Vec2> force;

    SoftSphereSimulation(
            Vec2 _boxSize,
            double _dt,
            double _velMag,
            size_t _accumulationSize
    ) : dt(_dt),
        velMag(_velMag),
        boxSize(_boxSize),
        accumulationSize(_accumulationSize) {}

    void initializeParticlesGrid(size_t, size_t);

    void calculateForce();

    void update();

    void outputParticlePositions(std::ostream &) const;
};

std::ostream &operator<<(std::ostream &, const SoftSphereSimulation &);

double &Vec2::operator[](size_t i) { return *(&x + i); }

double Vec2::operator[](size_t i) const { return *(&x + i); }

inline Vec2 Vec2::operator+(const Vec2 &v) const { return {x + v.x, y + v.y}; }

inline Vec2 Vec2::operator-(const Vec2 &v) const { return {x - v.x, y - v.y}; }

inline Vec2 Vec2::operator*(const Vec2 &v) const { return {x * v.x, y * v.y}; }

inline Vec2 Vec2::operator/(const Vec2 &v) const { return {x / v.x, y / v.y}; }

inline Vec2 Vec2::operator+(double a) const { return {x + a, y + a}; }

inline Vec2 Vec2::operator-(double a) const { return {x - a, y - a}; }

inline Vec2 Vec2::operator*(double a) const { return {x * a, y * a}; }

inline Vec2 Vec2::operator/(double a) const { return {x / a, y / a}; }

inline Vec2 Vec2::operator-() const { return {-x, -y}; }

inline Vec2 &Vec2::operator+=(const Vec2 &v) {
    x += v.x;
    y += v.y;
    return *this;
}

inline Vec2 &Vec2::operator-=(const Vec2 &v) {
    x -= v.x;
    y -= v.y;
    return *this;
}

inline Vec2 &Vec2::operator*=(double a) {
    x *= a;
    y *= a;
    return *this;
}

inline Vec2 &Vec2::operator/=(double a) {
    x /= a;
    y /= a;
    return *this;
}

inline Vec2 Vec2::operator==(const Vec2 &v) const { return {double(x == v.x), double(y == v.y)}; }

inline Vec2 Vec2::operator!=(const Vec2 &v) const { return {double(x != v.x), double(y != v.y)}; }

inline Vec2 Vec2::operator<(const Vec2 &v) const { return {double(x < v.x), double(y < v.y)}; }

inline Vec2 Vec2::operator>(const Vec2 &v) const { return {double(x > v.x), double(y > v.y)}; }

inline Vec2 Vec2::operator<=(const Vec2 &v) const { return {double(x <= v.x), double(y <= v.y)}; }

inline Vec2 Vec2::operator>=(const Vec2 &v) const { return {double(x >= v.x), double(y >= v.y)}; }

inline double Vec2::norm() const { return std::sqrt(norm2()); }

inline double Vec2::norm2() const { return x * x + y * y; }

inline double Vec2::dot(const Vec2 &v) const { return x * v.x + y * v.y; }

inline double Vec2::cross(const Vec2 &v) const { return x * v.y - y * v.x; }

Vec2 Vec2::unitRand() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution d(0., 1.);
    Vec2 rand{d(gen), d(gen)};
    while (rand.norm2() == 0.) {
        rand = {d(gen), d(gen)};
    }
    return rand / rand.norm();
}

inline Vec2 operator+(double a, const Vec2 &v) { return v + a; }

inline Vec2 operator-(double a, const Vec2 &v) { return -v + a; }

inline Vec2 operator*(double a, const Vec2 &v) { return v * a; }

inline Vec2 operator/(double a, const Vec2 &v) { return {a / v.x, a / v.y}; }

std::ostream &operator<<(std::ostream &os, const Vec2 &v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

void SoftSphereSimulation::initializeParticlesGrid(size_t n_x, size_t n_y) {
    nParticles = n_x * n_y;
    pos.resize(nParticles);
    vel.resize(nParticles);
    force.resize(nParticles);

    Vec2 nVec = {double(n_x), double(n_y)};
    Vec2 COMVelocity = {0., 0.};
    for (size_t x = 0; x < n_x; x++) {
        for (size_t y = 0; y < n_y; y++) {
            auto coord = Vec2({double(x), double(y)});

            size_t i = x + (n_x * y);
            pos[i] = ((coord + 0.5) / nVec - 0.5) * boxSize;

            vel[i] = velMag * Vec2::unitRand();
            COMVelocity += vel[i];

            force[i].x = force[i].y = 0;
        }
    }
    COMVelocity /= double(nParticles);
    for (size_t i = 0; i < nParticles; i++) {
        vel[i] -= COMVelocity;
    }
}

void SoftSphereSimulation::calculateForce() {
    double potential = 0;

// #pragma omp parallel for default(none)
    for (auto &i: force)
        i.x = i.y = 0;

    const double rc2 = std::pow(2., 1. / 3.);
// #pragma omp parallel for collapse(2) reduction(+:potential) default(none) shared(rc2)
    for (size_t i = 0; i < nParticles; ++i) {
        for (size_t j = 0; j < i; ++j) {
            Vec2 rij = pos[i] - pos[j];
            // Perform periodic boundary condition with minimum image convention
            rij -= (rij > 0.5 * boxSize) * boxSize - // Subtract boxSize if rij > 0.5 * boxSize for relevant coordinates
                   (rij < -0.5 * boxSize) * boxSize; // Add boxSize if rij < -0.5 * boxSize for relevant coordinates

            Vec2 dirVec = pos[i] - pos[j];
            double dist2 = dirVec.norm2();
            if (dist2 > rc2) continue;

            double distToTheNeg6 = std::pow(dist2, -3);
            Vec2 temp_force = dirVec * (48 * distToTheNeg6 * (distToTheNeg6 - 0.5) / dist2);
            potential += 4 * distToTheNeg6 * (distToTheNeg6 - 1.) + 1.;
            force[i] += temp_force; // add to i
            force[j] -= temp_force; // subtract from j
        }
    }

    macrostate.potential = potential;
}

void SoftSphereSimulation::update() {
    calculateForce();
    centerOfMass.x = centerOfMass.y = 0;
    // Kick-Drift-Kick scheme (Leapfrog)
// #pragma omp parallel for default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        vel[i] += force[i] * dt / 2.;
        pos[i] += vel[i] * dt;
    }

    double centerOfMassX = 0;
    double centerOfMassY = 0;
// #pragma omp parallel for reduction(+:centerOfMassX, centerOfMassY) default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        centerOfMassX += pos[i].x;
        centerOfMassX += pos[i].y;
    }
    centerOfMass = {centerOfMassX, centerOfMassY};
    centerOfMass /= double(nParticles);

    time += dt / 2.0;
    calculateForce();

// #pragma omp parallel for default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        vel[i] += force[i] * dt / 2.;
    }


    macrostate.kinetic = 0;
    totalMomentum.x = totalMomentum.y = 0;

    double totalMomentumX = 0;
    double totalMomentumY = 0;
    double kinetic = 0;
// #pragma omp parallel for reduction(+:totalMomentumX, totalMomentumY, kinetic) default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        totalMomentumX += vel[i].x;
        totalMomentumY += vel[i].y;
        kinetic += 0.5 * (vel[i].norm2());
    }
    totalMomentum = {totalMomentumX, totalMomentumY};
    centerOfMassVelocity = totalMomentum / double(nParticles);
    macrostate.kinetic = kinetic;

// #pragma omp parallel for default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        vel[i] -= centerOfMassVelocity;
        pos[i] -= centerOfMass;
    }

    double avg_kinetic = macrostate.kinetic / double(nParticles);
    double posForceDotProduct = 0;
// #pragma omp parallel for reduction(+:posForceDotProduct) default(none)
    for (size_t i = 0; i < nParticles; ++i) {
        posForceDotProduct += force[i].dot(pos[i]);
    }

    double temperature = 2. * avg_kinetic / 2.;
    double pressure = (temperature * double(nParticles) + posForceDotProduct / 2.) / (boxSize.x * boxSize.y);

    macrostate.temperature.push_back(temperature);
    macrostate.pressure.push_back(pressure);

    macrostate.temperature_acc.mean = macrostate.temperature_acc.variance = 0;
    macrostate.pressure_acc.mean = macrostate.pressure_acc.variance = 0;

    for (size_t i = macrostate.temperature.size() - 1, n = 0; i > 0 && n < accumulationSize; i--, n++) {
        macrostate.temperature_acc.mean += macrostate.temperature[i];
        macrostate.pressure_acc.mean += macrostate.pressure[i];
    }

    for (size_t i = macrostate.temperature.size() - 1, n = 0; i > 0 && n < accumulationSize; i--, n++) {
        macrostate.temperature_acc.variance += macrostate.temperature[i] * macrostate.temperature[i];
        macrostate.pressure_acc.variance += macrostate.pressure[i] * macrostate.pressure[i];
    }

    macrostate.temperature_acc.mean /= double(std::min(accumulationSize, macrostate.temperature.size()));
    macrostate.pressure_acc.mean /= double(std::min(accumulationSize, macrostate.pressure.size()));

    macrostate.temperature_acc.variance /= double(std::min(accumulationSize, macrostate.temperature.size()));
    macrostate.pressure_acc.variance /= double(std::min(accumulationSize, macrostate.pressure.size()));

    macrostate.temperature_acc.variance -= macrostate.temperature_acc.mean * macrostate.temperature_acc.mean;
    macrostate.pressure_acc.variance -= macrostate.pressure_acc.mean * macrostate.pressure_acc.mean;

    macrostate.temperature_acc.variance = std::max(macrostate.temperature_acc.variance, 0.);
    macrostate.pressure_acc.variance = std::max(macrostate.pressure_acc.variance, 0.);

    time += dt / 2.0;
}

void SoftSphereSimulation::outputParticlePositions(std::ostream &os) const {
    os << "Time: " << time << "\n";
    for (auto const &i : pos) {
        os << i.x << '\t' << i.y << '\n';
    }
}

std::ostream &operator<<(std::ostream &out, const SoftSphereSimulation &sim) {
    // Time | Total Energy | Total Kinetic Energy | Total Potential Energy | Pressure | Pressure StdDev | Temperature | Temperature StdDev | Total Momentum
    out.precision(std::numeric_limits<double>::max_digits10);
    out << sim.time << "\t"
        << (sim.macrostate.kinetic + sim.macrostate.potential) << "\t"
        << sim.macrostate.kinetic << "\t"
        << sim.macrostate.potential << "\t"
        << sim.macrostate.pressure_acc.mean << "\t"
        << std::sqrt(sim.macrostate.pressure_acc.variance) << "\t"
        << sim.macrostate.temperature_acc.mean << "\t"
        << std::sqrt(sim.macrostate.temperature_acc.variance) << "\t"
        << sim.totalMomentum.norm() << "\n";
    return out;
}

/**
 * We'll do our simulation with reduced units
 * Simulation Units:
 * Time: sigma * sqrt(mass / epsilon)
 * Length: sigma
 * Mass: mass
 * Energy: epsilon
 * Force: epsilon / sigma
 * Momentum: mass * sigma / (sigma * sqrt(mass / epsilon)) = sqrt(epsilon * mass)
 * Pressure: epsilon / sigma^3
 * Temperature: epsilon / (Boltzmann Constant)
 */
int main(int argc, char *argv[]) {
    size_t n_x;
    size_t n_y;
    double dt;
    double density;
    double velMag;
    size_t tMax = std::numeric_limits<size_t>::max();
    size_t accumulationSize;

    std::ostream *output;
    std::ostream *particleOutput;
    switch (argc) {
        case 3:
            particleOutput = new std::ofstream(argv[2]);
            output = new std::ofstream(argv[1]);
            break;
        case 2:
            particleOutput = new std::ofstream("/dev/null");
            output = new std::ofstream(argv[1]);
            break;
        case 1:
            particleOutput = new std::ofstream("/dev/null");
            output = &std::cout;
            break;
        default:
            std::cerr << "\033[1;31mInvalid number of arguments.\033[0m\n";
            return 1;
    }
    std::ostream &out = *output;
    std::ostream &particleOut = *particleOutput;


    // Input parameters
    std::cout << "Enter number of particles in x direction: ";
    std::cin >> n_x;
    std::cout << "Enter number of particles in y direction: ";
    std::cin >> n_y;
    std::cout << "Enter timestamp size: ";
    std::cin >> dt;
    std::cout << "Enter density: ";
    std::cin >> density;
    std::cout << "Enter initial velocity magnitude: ";
    std::cin >> velMag;
    std::cout << "Enter timestamp count: ";
    std::cin >> tMax;
    std::cout << "Enter accumulation step count: ";
    std::cin >> accumulationSize;

    Vec2 boxSize = {
            double(n_x) / std::sqrt(density),
            double(n_y) / std::sqrt(density)
    };

    SoftSphereSimulation sim(boxSize, dt, velMag, accumulationSize);
    sim.initializeParticlesGrid(n_x, n_y);

    out << "Time\t"
           "Total Energy\t"
           "Total Kinetic Energy\t"
           "Total Potential Energy\t"
           "Pressure\t"
           "Pressure StdDev\t"
           "Temperature\t"
           "Temperature StdDev\t"
           "Total Momentum\n";

    sim.outputParticlePositions(particleOut);
    for (size_t n = 0; n < tMax; n++) {
        sim.update();
        out << sim;
        sim.outputParticlePositions(particleOut);

        printf(
                "Timestamp %ld:\n"
                "\tTotal Energy: %e (Kinetic: %e, Potential: %e)\n"
                "\tTotal Momentum: %e\n"
                "\tCenter of Mass Velocity: %e\n"
                "\tPressure: %e ± %e\n"
                "\tTemperature: %e ± %e\n",
                n,
                sim.macrostate.kinetic + sim.macrostate.potential,
                sim.macrostate.kinetic,
                sim.macrostate.potential,
                sim.totalMomentum.norm(),
                sim.centerOfMassVelocity.norm(),
                sim.macrostate.pressure_acc.mean,
                std::sqrt(sim.macrostate.pressure_acc.variance),
                sim.macrostate.temperature_acc.mean,
                std::sqrt(sim.macrostate.temperature_acc.variance)
        );
    }
    out << std::endl;
    particleOut << std::endl;
    std::cout << std::endl;
}
