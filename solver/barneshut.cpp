#include "barneshut.h"

int select(float x, float y) {
    int a = x > 0 ? 1 : 0;
    if (y > 0) {
        a += 2;
    }
    return a;
}

float sign(float x) {
    return (x > 0.0f) ? 1.0f : -1.0f;
}

void insert(const float& x, const float& y, const float& G, const float& h, Branch& branch, const float xm, const float ym, const float D) {
    if (D * D < 0.01f * h) {
        float dx = x - xm;
        float dy = y - ym;
        branch.x += dx * G;
        branch.y += dy * G;
        branch.G += G;
        return;
    }
    if (branch.child == NULL) {
        branch.child = new Branch[4];
        float dx = branch.x - xm;
        float dy = branch.y - ym;
        int s = select(dx, dy);
        branch.child[s].x = branch.x;
        branch.child[s].y = branch.y;
        branch.child[s].G = branch.G;
        branch.x = dx * branch.G;
        branch.y = dy * branch.G;
    }
    float dx = x - xm;
    float dy = y - ym;
    branch.x += dx * G;
    branch.y += dy * G;
    branch.G += G;
    int s = select(dx, dy);
    if (isnan(branch.child[s].G)) {
        branch.child[s].x = x;
        branch.child[s].y = y;
        branch.child[s].G = G;
        return;
    }
    insert(x, y, G, h, branch.child[s], xm + sign(dx) * 0.25f * D, ym + sign(dy) * 0.25f * D, D * 0.5f);
    return;
}

void evaluate(float& u, float& v, const float& x, const float& y, const float& h, Branch branch, const float xm, const float ym, const float D) {
    if (branch.child == NULL) {
        float dx = x - branch.x;
        float dy = y - branch.y;
        float d2 = dx * dx + dy * dy + h;
        u -= branch.G * dy / d2;
        v += branch.G * dx / d2;
        return;
    }
    float dx = x - xm;
    float dy = y - ym;
    float d2 = dx * dx + dy * dy;
    if (d2 > (3.0f * 3.0f * D * D)) {
        float up = -2.0f * dx * dy * branch.x;
        up -= (-dx * dx + dy * dy - h) * branch.y;
        up /= d2 + h;
        up -= branch.G * dy;
        up /= d2 + h;

        float vp = 2.0f * dx * dy * branch.y;
        vp -= (-dx * dx + dy * dy + h) * branch.x;
        vp /= d2 + h;
        vp += branch.G * dx;
        vp /= d2 + h;

        u += up;
        v += vp;
        return;
    }
    //float up = 0.0f, vp = 0.0f;
    for (int i = 0; i < 4; i++) {
        if (isnan(branch.child[i].G)) continue;
        float dir_x = 0.5f * (i % 2) - 0.25f;
        float dir_y = 0.5f * (i / 2) - 0.25f;
        evaluate(u, v, x, y, h, branch.child[i], xm + dir_x * D, ym + dir_y * D, 0.5f * D);
    }
    return;
}

void deleteBranch(Branch& branch) {
    if (branch.child == NULL) return;
    for (int i = 0; i < 4; i++) {
        deleteBranch(branch.child[i]);
    }
    delete branch.child;
    return;
}

Tree createTree(const std::vector<float>& x, const std::vector<float>& G, const float& h) {
    float xmax = x[0], xmin = x[0], ymax = x[1], ymin = x[1];
    for (size_t i = 1; i < G.size(); i++) {
        xmax = x[2 * i + 0] > xmax ? x[2 * i + 0] : xmax;
        xmin = x[2 * i + 0] < xmin ? x[2 * i + 0] : xmin;
        ymax = x[2 * i + 1] > ymax ? x[2 * i + 1] : ymax;
        ymin = x[2 * i + 1] < ymin ? x[2 * i + 1] : ymin;
    }
    Tree tree = Tree(0.5f * (xmax + xmin), 0.5f * (ymax + ymin), ((xmax - xmin) > (ymax - ymin)) ? (xmax - xmin) : (ymax - ymin));
    tree.root->x = x[0];
    tree.root->y = x[1];
    tree.root->G = G[0];
    for (size_t i = 1; i < G.size(); i++) {
        insert(x[2 * i + 0], x[2 * i + 1], G[i], h, *tree.root, tree.xm, tree.ym, tree.D);
    }
    return tree;
}

void destroyTree(Tree& tree) {
    deleteBranch(*tree.root);
    delete tree.root;
}

void calcVelandStep(std::vector<float>& u, std::vector<float>& x, const std::vector<float>& G, const float& h, Tree& tree, const float& dt) {
#pragma omp parallel for
    for (long long i = 0; i < G.size(); i++) {
        float up = -1.0f, vp = 0.0f;
        evaluate(up, vp, x[2 * i + 0], x[2 * i + 1], h, *tree.root, tree.xm, tree.ym, tree.D);
        u[2 * i + 0] = up;
        u[2 * i + 1] = vp;
        x[2 * i + 0] += dt * up;
        x[2 * i + 1] += dt * vp;
    }
}

void calcVelRK2(std::vector<float>& u, std::vector<float>& x, const std::vector<float>& G, const float& h, const float& dt) {
    std::vector<float> x_p(x.size());
    Tree tree = createTree(x, G, h);
#pragma omp parallel for
    for (long long i = 0; i < G.size(); i++) {
        float up = -1.0f, vp = 0.0f;
        evaluate(up, vp, x[2 * i + 0], x[2 * i + 1], h, *tree.root, tree.xm, tree.ym, tree.D);
        x_p[2 * i + 0] = x[2 * i + 0] + 0.5f * dt * up;
        x_p[2 * i + 1] = x[2 * i + 1] + 0.5f * dt * vp;
    }
    destroyTree(tree);
    tree = createTree(x_p, G, h);
#pragma omp parallel for
    for (long long i = 0; i < G.size(); i++) {
        float up = -1.0f, vp = 0.0f;
        evaluate(up, vp, x_p[2 * i + 0], x_p[2 * i + 1], h, *tree.root, tree.xm, tree.ym, tree.D);
        x[2 * i + 0] += dt * up;
        x[2 * i + 1] += dt * vp;
        u[2 * i + 0] = up;
        u[2 * i + 1] = vp;
    }
    destroyTree(tree);
}
