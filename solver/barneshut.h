#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <string>
#include "omp.h"

struct Branch {
    float x, y, G = NAN;
    Branch* child = NULL;
};

struct Tree {
    float xm, ym, D;
    Branch* root;
    Tree(float xmc, float ymc, float Dc) {
        xm = xmc; ym = ymc; D = Dc;
        root = new Branch;
    }
};

void insert(const float& x, const float& y, const float& G, const float& h, Branch& branch, const float xm, const float ym, const float D);

void evaluate(float& u, float& v, const float& x, const float& y, const float& h, Branch branch, const float xm, const float ym, const float D);

void deleteBranch(Branch& branch);

Tree createTree(std::vector<float>& x, std::vector<float>& G, const float& h);

void destroyTree(Tree& tree);

void calcVelandStep(std::vector<float>& u, std::vector<float>& x, const std::vector<float>& G, const float& h, Tree& tree, const float& dt);

void calcVelRK2(std::vector<float>& u, std::vector<float>& x, const std::vector<float>& G, const float& h, const float& dt);
