#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "solver.h"

static const int directions[8][2] = {
    {0, 1},   // right
    {1, 0},   // dozn
    {0, -1},  // left
    {-1, 0},  // up
    {1, 1},   // diagonal down-right
    {1, -1},  // diagonal down-left
    {-1, 1},  // diagonal up-right
    {-1, -1}  // diagonal up-letf
};

int load_grid(const char *filename, Grid *g){
    FILE *file = fopen(filename, "r");
    if(!file){
        fprintf(stderr, "Error: cannot open file '%s'\n", filename);
        return 0;
    } 

    char buff[1024];
    g->rows = 0;
    g->cols = 0;

    while(fgets(buff, 1024, file)){
        int len = (int)strlen(buff);

       /* strip '\n' and '\r' first, so len is the real line length */
        while (len > 0 && (buff[len - 1] == '\n' || buff[len - 1] == '\r')){
            buff[--len] = '\0';
        }

        /* skip blank lines (checked after stripping) */
        if (len == 0) continue;

        if (len > MAX_SIZE || g->rows >= MAX_SIZE)
        {
            fprintf(stderr, "Error: grid too big in '%s'\n", filename);
            fclose(file);
            return 0;
        }
        
        if(g->cols != 0 && len != g->cols){
            fprintf(stderr, "Error: file line of different lenght '%s'\n", filename);
            fclose(file);
            return 0;
        }

        g->cols = len;
        for(int i = 0; i < len; i++){
            g->grid[g->rows][i] = (char)toupper((unsigned char)buff[i]);
        }
        g->rows++;
    }
    fclose(file);

    /* empty file or only blank lines: nothing to search in */
    if (g->rows == 0)
    {
        fprintf(stderr, "Error: no grid found in '%s'\n", filename);
        return 0;
    }

    return 1;
}

int check(const Grid *g, const char *word, int row, int col, int d){
    int len = (int)strlen(word);

    int dr = directions[d][0]; // row change per step 
    int dc = directions[d][1]; // column change per step 

    // Check whether the last letter is in the boundaries
    int end_row = row + dr*(len-1);
    int end_col = col + dc*(len-1);
    if((end_row < 0 || end_row >= g->rows) || (end_col < 0 || end_col >= g->cols)){
        return 0; // word bigger than the grid
    }

    // Go through the word and check if the word matches
    for(int i = 0; i < len; i++){
        int r = row + i*dr;
        int c = col + i*dc;

        //check if each character matches using toupper for extra assurance
        if(g->grid[r][c] != (char)toupper(word[i])){
            return 0; // Characters do not match
        }
    }
    // Word fully matches and return 1
    return 1;
}

/* out = {x0, y0, x1, y1}, with x = column and y = row */
int solve(const Grid *g, const char *word, int out[4]){

    int len = (int)strlen(word);
    if (len == 0) return 0;
    char first_letter = (char)toupper((unsigned char)word[0]); //Get first letter to start check from
    for(int i = 0; i < g->rows; i++){ // Go through row
        for(int j = 0; j < g->cols; j++){ // Go through column
            if(g->grid[i][j] == first_letter){ //Check for the first of word
                for(int d = 0; d < 8; d++) {
                    if(check(g, word, i, j, d)){ 
                        // Check if word is true, if yes store the start and end positions in out
                        int dr = directions[d][0];
                        int dc = directions[d][1];
                        out[0] = j;
                        out[1] = i;
                        out[2] = j + dc*(len-1);
                        out[3] = i + dr*(len-1);
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv){
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <grid_file> <word>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    const char *word = argv[2];

    if (word[0] == '\0')
    {
        fprintf(stderr, "Error: the word must not be empty\n");
        return EXIT_FAILURE;
    }

    Grid g;
    if (!load_grid(filename, &g)) return EXIT_FAILURE;

    if (g.rows < 5 || g.cols < 5) {
        fprintf(stderr, "Error: grid must be at least 5x5.\n");
        return EXIT_FAILURE;
    }

    int out[4];
    if(solve(&g, word, out)){
        printf("(%d,%d)(%d,%d)\n", out[0], out[1], out[2], out[3]);
    } 
    else{
        printf("Not Found\n");
    }
    return EXIT_SUCCESS;
}

