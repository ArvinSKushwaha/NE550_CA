#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using size_t = std::size_t;


struct Vec2 {
    double x, y;

    double& operator[](size_t i);
    double operator[](size_t i) const;

    Vec2 operator+(const Vec2& v) const;
    Vec2 operator-(const Vec2& v) const;
    Vec2 operator*(const Vec2& v) const;
    Vec2 operator/(const Vec2& v) const;

    Vec2 operator+(double a) const;
    Vec2 operator-(double a) const;
    Vec2 operator*(double a) const;
    Vec2 operator/(double a) const;
    Vec2 operator-() const;

    Vec2& operator+=(const Vec2& v);
    Vec2& operator-=(const Vec2& v);
    Vec2& operator*=(double a);
    Vec2& operator/=(double a);

    Vec2 operator==(const Vec2& v) const;
    Vec2 operator!=(const Vec2& v) const;
    Vec2 operator<(const Vec2& v) const;
    Vec2 operator>(const Vec2& v) const;
    Vec2 operator<=(const Vec2& v) const;
    Vec2 operator>=(const Vec2& v) const;

    double norm() const;
    double norm2() const;
    double dot(const Vec2& v) const;
    double cross(const Vec2& v) const;

    static Vec2 unitRand();
};

Vec2 operator+(double a, const Vec2& v);
Vec2 operator-(double a, const Vec2& v);
Vec2 operator*(double a, const Vec2& v);
Vec2 operator/(double a, const Vec2& v);

std::ostream& operator<<(std::ostream& os, const Vec2& v);

struct Macrostates {
    double kinetic, potential;
    double pressure, temperature;
};

class SoftSphereSimulation {
    public:
        double time = 0;
        double mass, sigma, epsilon, rc2, dt, velMag;
        Vec2 boxSize;
        Macrostates macrostate;
        double furthestDistance;
        Vec2 totalForce, totalMomentum, centerOfMass, centerOfMassVelocity;
        std::vector<Vec2> pos;
        std::vector<Vec2> vel;
        std::vector<Vec2> force;

        SoftSphereSimulation(
            double _mass,
            double _sigma,
            double _epsilon,
            Vec2 _boxSize,
            double _dt,
            double _velMag
        ) : mass(_mass),
            sigma(_sigma),
            epsilon(_epsilon),
            rc2(_sigma * _sigma * std::pow(2., 1./3.)),
            dt(_dt),
            velMag(_velMag),
            boxSize(_boxSize) {}

        void initialize_particles_grid(size_t n_x, size_t n_y);
        void calculate_force();
        void update();
};

std::ostream& operator<<(std::ostream&, const SoftSphereSimulation&);

double& Vec2::operator[] (size_t i) { return *(&x + i); }
double Vec2::operator[] (size_t i) const { return *(&x + i); }

Vec2 Vec2::operator+(const Vec2& v) const { return {x + v.x, y + v.y}; }
Vec2 Vec2::operator-(const Vec2& v) const { return {x - v.x, y - v.y}; }
Vec2 Vec2::operator*(const Vec2& v) const { return {x * v.x, y * v.y}; }
Vec2 Vec2::operator/(const Vec2& v) const { return {x / v.x, y / v.y}; }

Vec2 Vec2::operator+(double a) const { return {x + a, y + a}; }
Vec2 Vec2::operator-(double a) const { return {x - a, y - a}; }
Vec2 Vec2::operator*(double a) const { return {x * a, y * a}; }
Vec2 Vec2::operator/(double a) const { return {x / a, y / a}; }
Vec2 Vec2::operator-() const { return {-x, -y}; }

Vec2& Vec2::operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
Vec2& Vec2::operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
Vec2& Vec2::operator*=(double a) { x *= a; y *= a; return *this; }
Vec2& Vec2::operator/=(double a) { x /= a; y /= a; return *this; }

Vec2 Vec2::operator==(const Vec2& v) const { return {double(x == v.x), double(y == v.y)}; }
Vec2 Vec2::operator!=(const Vec2& v) const { return {double(x != v.x), double(y != v.y)}; }
Vec2 Vec2::operator<(const Vec2& v) const { return {double(x < v.x), double(y < v.y)}; }
Vec2 Vec2::operator>(const Vec2& v) const { return {double(x > v.x), double(y > v.y)}; }
Vec2 Vec2::operator<=(const Vec2& v) const { return {double(x <= v.x), double(y <= v.y)}; }
Vec2 Vec2::operator>=(const Vec2& v) const { return {double(x >= v.x), double(y >= v.y)}; }

double Vec2::norm() const { return std::sqrt(norm2()); }
double Vec2::norm2() const { return x * x + y * y; }
double Vec2::dot(const Vec2& v) const { return x * v.x + y * v.y; }
double Vec2::cross(const Vec2& v) const { return x * v.y - y * v.x; }

Vec2 Vec2::unitRand() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> d(0., 1.);
    Vec2 rand;
    do {
        rand = {d(gen), d(gen)};
    } while (rand.norm2() == 0.);
    return rand / rand.norm();
}

Vec2 operator+(double a, const Vec2& v) { return v + a; }
Vec2 operator-(double a, const Vec2& v) { return -v + a; }
Vec2 operator*(double a, const Vec2& v) { return v * a; }
Vec2 operator/(double a, const Vec2& v) { return { a / v.x, a / v.y}; }

std::ostream& operator<<(std::ostream& os, const Vec2& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

void SoftSphereSimulation::initialize_particles_grid(size_t n_x, size_t n_y) {
    size_t n_particles = n_x * n_y;
    pos.resize(n_particles);
    vel.resize(n_particles);
    force.resize(n_particles);

    Vec2 nVec = {double(n_x), double(n_y)};
    Vec2 COMVelocity = {0., 0.};
    for (size_t x = 0; x < n_x; x++) {
        for (size_t y = 0; y < n_y; y++) {
            Vec2 coord = Vec2({double(x), double(y)});

            size_t i = x + (n_x * y);
            pos[i] = ((coord + 0.5) / nVec - 0.5) * boxSize;

            vel[i] = velMag * Vec2::unitRand();
            COMVelocity += vel[i] * mass;

            force[i].x = force[i].y = 0;
        }
    }
    COMVelocity /= mass * n_particles;
    for (size_t i = 0; i < n_particles; i++) {
        vel[i] -= COMVelocity;
    }
}

void SoftSphereSimulation::calculate_force() {
    macrostate.potential = 0;
    totalForce.x = totalForce.y = 0;

    for (size_t i = 0; i < force.size(); i++)
        force[i].x = force[i].y = 0;

    for (size_t i = 0; i < force.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            Vec2 rij = pos[i] - pos[j];
            // Perform periodic boundary condition with minimum image convention
            rij -= (rij > 0.5 * boxSize) * boxSize - // Subtract boxSize if rij > 0.5 * boxSize for relevant coordinates
                   (rij < -0.5 * boxSize) * boxSize; // Add boxSize if rij < -0.5 * boxSize for relevant coordinates

            double dist2 = (pos[i] - pos[j]).norm2();
            if (dist2 > rc2) continue;

            double sigma2 = sigma * sigma;
            Vec2 temp_force = (pos[i] - pos[j]) * 48 * epsilon / sigma2 * (powf(sigma2 / dist2, 7) - 0.5 * powf(sigma2 / dist2, 4));
            force[i] += temp_force; // add to i
            force[j] -= temp_force; // subtract from j

            macrostate.potential += 4 * epsilon * (powf(sigma2 / dist2, 6) - powf(sigma2 / dist2, 3)) + epsilon;
        }
    }

    for (size_t i = 0; i < force.size(); ++i)
        totalForce += force[i];
}

void SoftSphereSimulation::update() {
    calculate_force();
    centerOfMass.x = centerOfMass.y = 0;
    furthestDistance = 0;
    // Kick-Drift-Kick scheme (Leapfrog)
    for (size_t i = 0; i < pos.size(); ++i) {
        vel[i] += force[i] * dt / (2. * mass);
        pos[i] += vel[i] * dt;
        centerOfMass += mass * pos[i];
        furthestDistance = std::max(furthestDistance, pos[i].norm2());
    }

    centerOfMass /= mass * pos.size();
    time += dt / 2.0;
    calculate_force();

    macrostate.kinetic = 0;
    totalMomentum.x = totalMomentum.y = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        vel[i] += force[i] * dt / (2. * mass);
        macrostate.kinetic += vel[i].norm2() * mass / 2.0;
        totalMomentum += vel[i] * mass;
    }
    centerOfMassVelocity = totalMomentum / (mass * pos.size());

    for (size_t i = 0; i < pos.size(); ++i) {
        vel[i] -= centerOfMassVelocity;
        pos[i] -= centerOfMass;
    }

    double avg_kinetic = macrostate.kinetic / pos.size();
    const double boltzmann = 0.83144; // Boltzmann constant in (amu * A^2 / ps^2)/K
    double posForceDotProduct = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        posForceDotProduct += force[i].dot(pos[i]);
    }

    macrostate.temperature = 2. * avg_kinetic / (2. * boltzmann);
    macrostate.pressure = (macrostate.temperature * pos.size() + posForceDotProduct / 2.) / (boxSize.x * boxSize.y);
    furthestDistance = std::sqrt(furthestDistance);

    time += dt / 2.0;
}

std::ostream& operator<<(std::ostream& out, const SoftSphereSimulation& sim) {
    // Time | Total Energy | Total Kinetic Energy | Total Potential Energy | Pressure | Temperature | COM (v) | COM Velocity (v) | Total Force (v) | Total Momentum (v)
    out << sim.time << "\t" 
        << (sim.macrostate.kinetic + sim.macrostate.potential) << "\t"
        << sim.macrostate.kinetic << "\t"
        << sim.macrostate.potential << "\t"
        << sim.macrostate.pressure * 1.6605e+7 << "\t"
        << sim.macrostate.temperature << "\t"
        << sim.centerOfMass << "\t"
        << sim.centerOfMassVelocity << "\t"
        << sim.totalForce << "\t"
        << sim.totalMomentum << "\n";
    return out;
}

/**
 * We'll do our simulation with units angstrom, amu, and picoseconds
 * Simulation Units:
 * Time: ps
 * Length: A = 0.1 nm
 * Mass: amu
 * Macrostates: amu * A^2 / ps^2 = 1.0364e-4 eV
 * Force: amu * A / ps^2 = 1.0364e-4 eV / A
 * Momentum: amu * A / ps
 * Pressure: amu / (A ps^2) = 1.0364e-4 eV / A^3 = 1.6605e+7 Pa
 * Temperature: K = Energy / k = Energy * 1.20272 K / (amu * A^2 / ps^2)
 * Boltzmann Constant: k = 0.83144 (amu * A^2 / ps^2) / K
 */
int main(int argc, char *argv[]) {
    size_t n_x, n_y;
    double dt, density, velMag, epsilon, sigma, mass, tMax = INFINITY;

    std::ostream *output;
    switch (argc) {
        case 2:
            output = new std::ofstream(argv[1]);
            break;
        case 1:
            output = &std::cout;
            break;
        default:
            std::cerr << "\033[1;31mInvalid number of arguments.\033[0m\n";
            return 1;
    }
    std::ostream &out = *output;


    // Input parameters
    std::cout << "Enter number of particles in x direction: ";
    std::cin >> n_x;
    std::cout << "Enter number of particles in y direction: ";
    std::cin >> n_y;
    std::cout << "Enter timestep size (ps): ";
    std::cin >> dt;
    std::cout << "Enter density (particles per angstrom^2): ";
    std::cin >> density;
    std::cout << "Enter initial velocity magnitude (A/ps): ";
    std::cin >> velMag;
    std::cout << "Enter epsilon (eV): ";
    std::cin >> epsilon;
    std::cout << "Enter sigma (nm): ";
    std::cin >> sigma;
    std::cout << "Enter mass (amu): ";
    std::cin >> mass;
    std::cout << "Enter tMax (ps): ";
    std::cin >> tMax;

    Vec2 boxSize = {
        n_x / std::sqrt(density),
        n_y / std::sqrt(density)
    };
    epsilon = epsilon / 1.0364e-4; // convert eV to amu * A^2 / ps^2
    sigma = sigma / 0.1; // convert nm to A

    SoftSphereSimulation sim(mass, sigma, epsilon, boxSize, dt, velMag);
    sim.initialize_particles_grid(n_x, n_y);

    out << "Time\tTotal Energy\tTotal Kinetic Energy\tTotal Potential Energy\tPressure\tTemperature\tCOM (v)\tCOM Velocity (v)\tTotal Force (v)\tTotal Momentum (v)\n";
    while (sim.time < tMax) {
        for (int i = 0; i < 100 && sim.time < tMax; i++) {
            sim.update();
            out << sim;
        }
        printf(
            "Timestep %f:\n"
            "\tTotal Energy: %e eV (Kinetic: %e eV, Potential: %e eV)\n"
            "\tTotal Momentum: [ %e, %e ] amu A / ps\n"
            "\tTotal Force: [ %e, %e ] amu A / ps^2\n"
            "\tCenter of Mass: [ %e, %e ] A\n"
            "\tCenter of Mass Velocity: [ %e, %e ] A / ps\n"
            "\tPressure: %e Pa\n"
            "\tTemperature: %e K\n"
            "\tFurthest Particle: %e A\n",
            sim.time,
            (sim.macrostate.kinetic + sim.macrostate.potential) * 1.0364e-4,
            sim.macrostate.kinetic * 1.0364e-4,
            sim.macrostate.potential * 1.0364e-4,
            sim.totalMomentum.x,
            sim.totalMomentum.y,
            sim.totalForce.x,
            sim.totalForce.y,
            sim.centerOfMass.x,
            sim.centerOfMass.y,
            sim.centerOfMassVelocity.x,
            sim.centerOfMassVelocity.y,
            sim.macrostate.pressure * 1.6605e+7,
            sim.macrostate.temperature,
            sim.furthestDistance
        );
    }
    out << std::endl;
}

