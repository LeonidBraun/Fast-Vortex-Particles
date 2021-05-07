#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <string>

#include "omp.h"
#include "vtu11/vtu11.hpp"
#include "barneshut.h"

void exportData(std::vector<float> x, std::vector<float> u, std::vector<float> G, std::string name, size_t iter) {
    std::vector<vtu11::VtkIndexType> connectivity(G.size());
    std::vector<double> points(G.size() * 3);
    std::vector<vtu11::VtkIndexType> offsets(G.size());
    std::vector<vtu11::VtkCellType> types(G.size());

    std::vector<double> pointU(G.size() * 3);
    std::vector<double> pointG(G.size());
    for (size_t i = 0; i < G.size(); i++) {
        connectivity[i] = i;
        points[3 * i + 0] = x[2 * i + 0];
        points[3 * i + 1] = x[2 * i + 1];
        points[3 * i + 2] = 0.0f;
        offsets[i] = i + 1;
        types[i] = 1;
        pointU[3 * i + 0] = u[2 * i + 0];
        pointU[3 * i + 1] = u[2 * i + 1];
        pointU[3 * i + 2] = 0.0f;
        pointG[i] = G[i];
    }

    vtu11::Vtu11UnstructuredMesh mesh{ points, connectivity, offsets, types };
    std::vector<vtu11::DataSetInfo> dataSetInfo{
        { "Velocity", vtu11::DataSetType::PointData, 3 },
        { "Zirkulation", vtu11::DataSetType::PointData, 1 },
    };
    vtu11::writeVtu(name + "_" + std::to_string(iter) + ".vtu", mesh, dataSetInfo, { pointU, pointG }, "RawBinary");
}

void force(std::vector<float>& u, std::vector<float>& x, std::vector<float>& G, const float &dt) {
    x.push_back(0); x.push_back(-0.5);
    x.push_back(0); x.push_back(-0.49);
    x.push_back(0); x.push_back(0.49);
    x.push_back(0); x.push_back(0.5);

    u.push_back(0); u.push_back(0);
    u.push_back(0); u.push_back(0);
    u.push_back(0); u.push_back(0);
    u.push_back(0); u.push_back(0);

    G.push_back(dt / 2);
    G.push_back(dt / 2);
    G.push_back(-dt / 2);
    G.push_back(-dt / 2);
}

void downsample(std::vector<float>& x, std::vector<float>& G, float dx, float xcut) {
    std::vector<long> cell(G.size());
    long idx_max = (long)(20 / dx);
#pragma omp parallel for
    for (long long i = 0; i < G.size(); i++) {
        long idx = ((x[2 * i + 1] + 10) / dx);
        cell[i] = (idx >= 0 && idx < idx_max && (x[2 * i + 0] < xcut) && (x[2 * i + 0] > (xcut-dx))) ? idx : -1;
    }

#pragma omp parallel for
    for (long id = 0; id < idx_max; id++) {
        size_t i_first = -1;
        float x_first = NAN;
        float y_first = NAN;
        float G_first = NAN;
        for (size_t i = 0; i < G.size(); i++) {
            if (cell[i] == id) {
                if (isnan(G_first)) {
                    i_first = i;
                    x_first = x[2 * i + 0] * G[i];
                    y_first = x[2 * i + 1] * G[i];
                    G_first = G[i];
                }
                else {
                    x_first += x[2 * i + 0] * G[i];
                    y_first += x[2 * i + 1] * G[i];
                    G_first += G[i];
                    G[i] = 0.0f;
                }
            }
        }
        if (!isnan(G_first)) {
            x[2 * i_first + 0] = x_first / G_first;
            x[2 * i_first + 1] = y_first / G_first;
            G[i_first] = G_first;
        }
    }
}

void merge(std::vector<float>& u, std::vector<float>& x, std::vector<float>& G) {
    downsample(x, G, 0.3f, -10.0f);
    downsample(x, G, 1.0f, -15.0f);
    downsample(x, G, 3.0f, -20.0f);
    downsample(x, G, 10.0f, -30.0f);
    

    std::vector<float> up;
    std::vector<float> xp;
    std::vector<float> Gp;
    for (long long i = 0; i<G.size(); i++) {
        if (std::fabsf(G[i]) < 1e-6) continue;
        xp.push_back(x[2 * i + 0]);
        xp.push_back(x[2 * i + 1]);
        up.push_back(u[2 * i + 0]);
        up.push_back(u[2 * i + 1]);
        Gp.push_back(G[i]);
    }
    u = up;
    x = xp;
    G = Gp;
}

void calcVelNaiveandStep(std::vector<float> &u, std::vector<float> &x, const std::vector<float> &G, const float &h, const float &dt) {
#pragma omp parallel for
    for (long long i = 0; i < G.size(); i++) {
        float U = 0, V = 0;
        float xi = x[2 * i + 0];
        float yi = x[2 * i + 1];
        for (size_t j = 0; j < G.size(); j++) {
            float dx = xi - x[2 * j + 0];
            float dy = yi - x[2 * j + 1];
            float d2 = dx * dx + dy * dy + h;
            U -= G[j] * dy / d2;
            V += G[j] * dx / d2;
        }
        u[2 * i + 0] = U-1;
        u[2 * i + 1] = V;
    }
#pragma omp parallel for
    for (long long j = 0; j < G.size(); j++) {
        x[2 * j + 0] += dt * (u[2 * j + 0]);
        x[2 * j + 1] += dt * (u[2 * j + 1]);
    }
}

int main() {

    float H = 0.01f;
    float h = H * H;
    std::vector<float> u;
    std::vector<float> x;
    std::vector<float> G;
    for (size_t i = 0; i < 200; i++) {
        float a = 2.0f*3.14f*(float)i / 200.0f;
        float X = std::cos(a)-5;
        float Y = std::sin(a);
        x.push_back(X);
        x.push_back(Y);
        G.push_back(Y*0);
        u.push_back(0);
        u.push_back(0);
    }
    float dt = 0.001f;

    std::cout << G.size() << "\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 400; i++) {
        exportData(x, u, G, "test", i);
        std::cout << "iteration: " << i << "\n";

        for (size_t l = 0; l < 50; l++) {
            force(u, x, G, 1 * dt);
            for (size_t k = 0; k < 1; k++) {
                //calcVelNaiveandStep(u, x, G, h, dt);

                Tree tree = createTree(x, G, h);
                calcVelandStep(u, x, G, h, tree, dt);
                destroyTree(tree);
            }
        }
        merge(u, x, G);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << "delta time: " << fp_ms.count() << "\n";

	return 0;
}