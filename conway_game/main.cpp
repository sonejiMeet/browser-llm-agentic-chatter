#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>

const int WIDTH = 30;
const int HEIGHT = 15;

void clearScreen() {
#ifdef _WIN32
system("cls");
#else
system("clear");
#endif
}

int countNeighbors(const std::vector<std::vector<int>>& grid, int x, int y) {
int count = 0;

for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;

        int nx = (x + dx + WIDTH) % WIDTH;
        int ny = (y + dy + HEIGHT) % HEIGHT;

        count += grid[ny][nx];
    }
}

return count;

}

void display(const std::vector<std::vector<int>>& grid) {
for (int y = 0; y < HEIGHT; y++) {
for (int x = 0; x < WIDTH; x++) {
std::cout << (grid[y][x] ? "O " : ". ");
}
std::cout << "\n";
}
}

int main() {
std::vector<std::vector<int>> grid(HEIGHT, std::vector<int>(WIDTH, 0));

// Glider
grid[1][2] = 1;
grid[2][3] = 1;
grid[3][1] = 1;
grid[3][2] = 1;
grid[3][3] = 1;

// Blinker
grid[8][10] = 1;
grid[8][11] = 1;
grid[8][12] = 1;

for (int generation = 1; generation <= 200; generation++) {
    clearScreen();

    std::cout << "Conway's Game of Life - Generation " 
              << generation << "\n\n";

    display(grid);

    std::vector<std::vector<int>> next = grid;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int neighbors = countNeighbors(grid, x, y);

            if (grid[y][x]) {
                next[y][x] = (neighbors == 2 || neighbors == 3);
            } else {
                next[y][x] = (neighbors == 3);
            }
        }
    }

    grid = next;

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

return 0;

}
