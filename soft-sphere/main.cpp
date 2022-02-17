#include "matrix.hpp"
#include "input_processing.hpp"


struct Vec3
{
    double x, y, z;
    double& operator[](int i);
};

inline double& Vec3::operator[] (int i)
{
    return *(&x + i);
}

inline Vec3 add(Vec3 a, Vec3 b)
{
    Vec3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

inline Vec3 mul(Vec3 a, double b)
{
    Vec3 c;
    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;
    return c;
}

inline Vec3 sub(Vec3 a, Vec3 b)
{
    return add(a, mul(b, -1));
}

inline Vec3 div(Vec3 a, double b)
{
    return mul(a, 1 / b);
}

inline double dot(Vec3 a, Vec3 b)
{
    double dot = 0;
    dot += a.x * b.x;
    dot += a.y * b.y;
    dot += a.z * b.z;
    return dot;
}

class HardBallSimulation {
    public:
        double mass, sigma, epsilon;
        double sigma_216, box_size;
        std::vector<Vec3> pos;
        std::vector<Vec3> vel;
        std::vector<Vec3> force;
        HardBallSimulation(double _mass, double _sigma, double _epsilon, double _box_size) : mass(_mass), sigma(_sigma), epsilon(_epsilon), sigma_216(_sigma * powf(2., 1./6.)), box_size(_box_size) {}
        void initialize_particles(size_t n_particles) {
            pos.resize(n_particles);
            vel.resize(n_particles);
            force.resize(n_particles);
            for (size_t i = 0; i < n_particles; i++) {
                pos[i].x = (2.0 * rand() / double(RAND_MAX) - 1.0) * box_size;
                pos[i].y = (2.0 * rand() / double(RAND_MAX) - 1.0) * box_size;
                pos[i].z = (2.0 * rand() / double(RAND_MAX) - 1.0) * box_size;
                force[i] = {0, 0, 0};
            }
        }
        void calculate_force() {
            for (size_t i = 0; i < force.size(); ++i) {
                force[i] = {0, 0, 0};
                for (size_t j = 0; j < i; ++j) {
                    double dist2 = dot(sub(pos[i], pos[j]), sub(pos[i], pos[j]));
                    if (dist2 > sigma_216 * sigma_216) continue;
                    double dist = sqrt(dist2);
                    Vec3 temp_force = mul(sub(pos[i], pos[j]), 48 * epsilon / (sigma * sigma) * (powf(sigma / dist2, 14) - 0.5 * powf(sigma / dist, 8)));
                    force[i] = add(force[i], temp_force);
                    force[j] = sub(force[j], temp_force);
                }
            }
        }
        double total_energy() {
            double energy = 0;
            for (size_t i = 0; i < force.size(); ++i) {
                energy += dot(vel[i], vel[i]) * mass / 2;
                for (size_t j = 0; j < i; ++j) {
                    double dist2 = dot(sub(pos[i], pos[j]), sub(pos[i], pos[j]));
                    if (dist2 > sigma_216 * sigma_216) continue;
                    double dist = sqrt(dist2);
                    energy += 4 * epsilon * (powf(sigma / dist, 12) - powf(sigma / dist2, 6)) + epsilon;
                }
            }
            return energy;
        }
        Vec3 total_momentum() {
            Vec3 momentum = {0, 0, 0};
            for (size_t i = 0; i < vel.size(); ++i) {
                momentum = add(momentum, mul(vel[i], mass));
            }
            return momentum;
        }
        void update(double dt) {
            for (size_t i = 0; i < pos.size(); ++i) {
                pos[i] = add(pos[i], 
                    add(
                        mul(vel[i], dt), 
                        mul(force[i], dt * dt / (2 * mass))
                    )
                );
                Vec3 a_prev = div(force[i], mass);
                calculate_force();
                Vec3 a_next = div(force[i], mass);
                vel[i] = add(vel[i], mul(add(a_next, a_prev), dt / 2.0));
            }
        }
};

int main() {
    // We'll do our simulation with units angstrom, amu, and picoseconds
    // For epsilon = 0.01034 eV and sigma = 0.34 nm for argon (Ar), we convert to:
    // m_argon = 39.948 amu
    // sigma = 0.34 nm = 3.4 A
    // epsilon = 0.01034 eV = 99.765 amu * A^2 / ps^2
    // dt = 0.001 ps = 1 fs
    // Simulation Units:
    // Time: ps
    // Length: A
    // Mass: amu
    // Energy: amu * A^2 / ps^2 = 1.0364e-4 eV
    // Force: amu * A / ps^2 = 1.0364e-4 eV / A
    HardBallSimulation sim(39.948, 3.4, 99.765, 100.0);
    sim.initialize_particles(1000); // Ideally, around 3178 spheres with radius (0.34 * 2^(1/6)) nm can fit in the box of size 100 A.
    while (true) {
        Vec3 momentum = sim.total_momentum();
        Vec3 max_vel = {0, 0, 0};
        for (size_t i = 0; i < sim.vel.size(); ++i) {
            if (dot(sim.vel[i], sim.vel[i]) > dot(max_vel, max_vel)) {
                max_vel = sim.vel[i];
            }
        }
        printf(
            "Energy: %e Momentum: [ %e, %e, %e ] Max Velocity: [ %e, %e, %e ]\n", 
            sim.total_energy(), 
            momentum.x, 
            momentum.y, 
            momentum.z, 
            max_vel.x, 
            max_vel.y, 
            max_vel.z
        );
        sim.calculate_force();
        sim.update(1e-3); // 1 fs
        
    }
}
