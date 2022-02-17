#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
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

    double norm2() const;
    double dot(const Vec2& v) const;
    double cross(const Vec2& v) const;
};

inline double& Vec2::operator[] (size_t i) { return *(&x + i); }
inline double Vec2::operator[] (size_t i) const { return *(&x + i); }

inline Vec2 Vec2::operator+(const Vec2& v) const { return {x + v.x, y + v.y}; }
inline Vec2 Vec2::operator-(const Vec2& v) const { return {x - v.x, y - v.y}; }
inline Vec2 Vec2::operator*(const Vec2& v) const { return {x * v.x, y * v.y}; }
inline Vec2 Vec2::operator/(const Vec2& v) const { return {x / v.x, y / v.y}; }

inline Vec2 Vec2::operator+(double a) const { return {x + a, y + a}; }
inline Vec2 Vec2::operator-(double a) const { return {x - a, y - a}; }
inline Vec2 Vec2::operator*(double a) const { return {x * a, y * a}; }
inline Vec2 Vec2::operator/(double a) const { return {x / a, y / a}; }
inline Vec2 Vec2::operator-() const { return {-x, -y}; }

inline Vec2& Vec2::operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
inline Vec2& Vec2::operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
inline Vec2& Vec2::operator*=(double a) { x *= a; y *= a; return *this; }
inline Vec2& Vec2::operator/=(double a) { x /= a; y /= a; return *this; }

inline Vec2 Vec2::operator==(const Vec2& v) const { return {double(x == v.x), double(y == v.y)}; }
inline Vec2 Vec2::operator!=(const Vec2& v) const { return {double(x != v.x), double(y != v.y)}; }
inline Vec2 Vec2::operator<(const Vec2& v) const { return {double(x < v.x), double(y < v.y)}; }
inline Vec2 Vec2::operator>(const Vec2& v) const { return {double(x > v.x), double(y > v.y)}; }
inline Vec2 Vec2::operator<=(const Vec2& v) const { return {double(x <= v.x), double(y <= v.y)}; }
inline Vec2 Vec2::operator>=(const Vec2& v) const { return {double(x >= v.x), double(y >= v.y)}; }

inline double Vec2::norm2() const { return x * x + y * y; }
inline double Vec2::dot(const Vec2& v) const { return x * v.x + y * v.y; }
inline double Vec2::cross(const Vec2& v) const { return x * v.y - y * v.x; }

inline Vec2 operator+(double a, const Vec2& v) { return v + a; }
inline Vec2 operator-(double a, const Vec2& v) { return -v + a; }
inline Vec2 operator*(double a, const Vec2& v) { return v * a; }
inline Vec2 operator/(double a, const Vec2& v) { return { a / v.x, a / v.y}; }

struct Energy {
    double kinetic, potential;
};

class HardBallSimulation {
    public:
        double time = 0;
        double mass, sigma, epsilon, rc2, dt;
        Vec2 boxSize;
        Energy energy;
        Vec2 totalForce;
        Vec2 totalMomentum;
        std::vector<Vec2> pos;
        std::vector<Vec2> vel;
        std::vector<Vec2> force;

        HardBallSimulation(
            double _mass, 
            double _sigma,
            double _epsilon, 
            Vec2 _boxSize,
            double _dt
        ) : mass(_mass), 
            sigma(_sigma), 
            epsilon(_epsilon), 
            rc2(_sigma * _sigma * powf(2., 1./3.)), 
            dt(_dt),
            boxSize(_boxSize) {}

        void initialize_particles(size_t n_particles);
        void initialize_particles_grid(size_t n_x, size_t n_y);
        void calculate_force();
        void update();
};

std::ostream& operator<<(std::ostream&, const HardBallSimulation&);

void HardBallSimulation::initialize_particles(size_t n_particles) {
    pos.resize(n_particles); vel.resize(n_particles); force.resize(n_particles);
    for (size_t i = 0; i < n_particles; i++) {
        pos[i] = Vec2({
                    (2.0 * rand() / double(RAND_MAX) - 1.0),
                    (2.0 * rand() / double(RAND_MAX) - 1.0)
                }) * boxSize;
        
        vel[i].x = vel[i].y = 0;
        force[i].x = force[i].y = 0;
    }
}

void HardBallSimulation::initialize_particles_grid(size_t n_x, size_t n_y) {
    size_t n_particles = n_x * n_y;
    pos.resize(n_particles); vel.resize(n_particles); force.resize(n_particles);

    for (size_t x = 0; x < n_x; x++) {
        for (size_t y = 0; y < n_y; y++) {
            size_t i = x + (n_x * y);
            pos[i] = Vec2({
                        (2.0 * x / double(n_x) - 1.0),
                        (2.0 * y / double(n_y) - 1.0)
                    }) * boxSize;

            vel[i].x = vel[i].y = 0;
            force[i].x = force[i].y = 0;
        }
    }
}

void HardBallSimulation::calculate_force() {
    energy.potential = 0;
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

            energy.potential += 4 * epsilon * (powf(sigma2 / dist2, 6) - powf(sigma2 / dist2, 3));
        }
    }

    for (size_t i = 0; i < force.size(); ++i)
        totalForce += force[i];
}

void HardBallSimulation::update() {
    calculate_force();
    // Kick-Drift-Kick scheme (Leapfrog)
    for (size_t i = 0; i < pos.size(); ++i) {
        vel[i] += force[i] * dt / (2. * mass);
        pos[i] += vel[i] * dt;
    }
    time += dt / 2.0;
    calculate_force();

    energy.kinetic = 0;
    totalMomentum.x = totalMomentum.y = 0;
    for (size_t i = 0; i < pos.size(); ++i) {
        vel[i] += force[i] * dt / (2. * mass);
        energy.kinetic += vel[i].norm2() * mass / 2.0;
        totalMomentum += vel[i] * mass;
    }
    time += dt / 2.0;
}

std::ostream& operator<<(std::ostream& out, const HardBallSimulation& sim) {
    out << "Timestep: " << sim.time << "\n";
    for (size_t i = 0; i < sim.pos.size(); i++) {
        out << "\t";
        out << sim.pos[i].x << " " << sim.pos[i].y << " ";
        out << sim.vel[i].x << " " << sim.vel[i].y << " ";
        out << sim.force[i].x << " " << sim.force[i].y << "\n"; 
    }
    return out;
}

/**
 * We'll do our simulation with units angstrom, amu, and picoseconds
 * For epsilon = 0.01034 eV and sigma = 0.34 nm for argon (Ar), we convert to:
 * m_argon = 39.948 amu
 * sigma = 0.34 nm = 3.4 A
 * epsilon = 0.01034 eV = 99.765 amu * A^2 / ps^2
 * dt = 0.001 ps = 1 fs
 * Simulation Units:
 * Time: ps
 * Length: A
 * Mass: amu
 * Energy: amu * A^2 / ps^2 = 1.0364e-4 eV
 * Force: amu * A / ps^2 = 1.0364e-4 eV / A
 * Momentum: amu * A / ps
 */
int main(int argc, char *argv[]) {
    HardBallSimulation sim(39.948, 3.4, 99.765, {100.0, 100.0}, 1e-3);
    sim.initialize_particles_grid(80, 80); // 80^2 = 6400 particles

    std::ostream *output = argc < 2 ? &std::cout : new std::ofstream(argv[1]);
    std::ostream &out = *output;
    double tMax = INFINITY;
    if (argc > 2) {
        tMax = std::atof(argv[2]);
    }

    out << sim;
    while (sim.time < tMax) {
        sim.update();
        out << sim;
        printf(
            "Timestep %f:\n"
            "\tTotal Energy: %e eV (Kinetic: %e eV, Potential: %e eV)\n"
            "\tTotal Momentum: [ %e, %e ] amu A / ps\n"
            "\tTotal Force: [ %e, %e ] amu A / ps^2\n",
            sim.time,
            (sim.energy.kinetic + sim.energy.potential) * 1.0364e-4,
            sim.energy.kinetic * 1.0364e-4,
            sim.energy.potential * 1.0364e-4,
            sim.totalMomentum.x, 
            sim.totalMomentum.y, 
            sim.totalForce.x,
            sim.totalForce.y
        );
    }
}
