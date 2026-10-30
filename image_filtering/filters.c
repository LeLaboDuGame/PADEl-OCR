#include "filters.h"

//fills hist with the grayscale histogram of src: hist[v] = number of pixels whose gray value is exactly v (v from 0 to 255)
void compute_histogram(const Image *src, int hist[256]){
    memset(hist, 0, 256*sizeof(int)); // a local array is not zero-initialized
    int total = src->w * src->h;
    for(int i = 0; i < total; i++){
        // The pixel value is used as an index
        hist[src->data[i]]++;
    }
}

//Otsu's method is about picking the threshold that best separates the image between dark and clear pixels
//for every threshold t (0-255), pxl with a value <= t form the dark grp and the rest form the clear grp
//each t is scored with a the between-class variance which is large when the two grps are far apart and low when similar in size
//the threshold is computed with the histogram of the image, making it adapt to the image instead of being a fixed value
int otsu_treshold(const Image *src){
    int hist[256];
    compute_histogram(src, hist);

    long total = (long)src->w * src->h;
    //sum of all grey values of all the pxls
    double sum = 0;
    for (int v = 0; v < 256; v++){
        sum += (double)v * hist[v];
    }
    // we put double and long bc int isn't big enough for the kind of calculus we have to perform on pixels
    double sum_b = 0; //sum of grey on dark grp 
    long w_b = 0; //nbr pxl dark grp
    double best_var = -1; //best of variances
    int best_t = 0; //best variance threshold

    for (int t = 0; t < 256; t++){
        w_b += hist[t];
        if (w_b == 0){
            continue; //case of no dark pixels yet, empty pixel grp
        }
        long w_f = total - w_b;
        if (w_f == 0){
            break; //case of no more clear pixels, we're finished
        }
        sum_b += (double)t * hist[t];
        //grey value average fo dark and clear grp
        double m_b = sum_b / w_b;
        double m_f = (sum - sum_b) / w_f;

        double var = (double)w_b * (double)w_f * (m_b - m_f) * (m_b - m_f); //computation of interclass variance 
        if (var > best_var){
            best_var = var;
            best_t = t;
        }
    }
    return best_t;
}

//Binarizes src with threshold t and returns the result as a new image with pxls <= t becoming black and the rest becoming clear
Image *treshold(const Image *src, int t){
    Image *new_img = image_create(src->w, src->h);
    if (!new_img){
        return NULL;
    }
    int total = src->w * src->h;
    for (int i = 0; i < total; i++){
        // Each pixel is processed independently, so a flat loop is enough
        if (src->data[i] <= t){
            new_img->data[i] = 0;
        } else {
            new_img->data[i] = 255;
        }
    }
    return new_img;
}

