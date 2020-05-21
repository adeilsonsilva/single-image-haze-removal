#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <cmath>

#define    W_ORIGINAL                      "Original"
#define    W_DARK_CHANNEL                  "Dark Channel"
#define    W_TRANSMISSION_IMAGE            "Transmission Image"
#define    W_REFINED_TRANSMISSION_IMAGE    "Refined Transmission Image"
#define    W_RESULT                        "De-hazed"

/**
 * @brief Get the dark channel prior object
 *
 * @param src       RGB image
 * @param w_size    Size of patch to consider (default is 15)
 *
 * @return cv::Mat  One-channel image with dark channel prior
 */
static cv::Mat get_dark_channel_prior(const cv::Mat &src, const int w_size=15)
{
    cv::Mat dark_channel = cv::Mat::zeros(src.size(), CV_64F);

    int win_center_distance = (w_size - 1) / 2;
    double min_value = 0;
    double max_value = 0;

    // Create padded cv::matrix with repeated values (we need to create a patch
    // centralized in every original pixel)
    int n_rows = src.rows;
    int n_cols = src.cols;
    int padding_size = (w_size % 2 == 0) ? w_size : w_size+1;
    cv::Mat padded_src (n_rows+padding_size,
                    n_cols+padding_size,
                    CV_64FC3);

    // fill above and below borders
    for (int k = 0; k < win_center_distance+1; k++)
    {
        // copy to central row
        src.row(0).copyTo(
            padded_src.row(k).colRange(win_center_distance+1, win_center_distance+n_cols+1)
        );
        // copy to whats to the left of the row (diagonal)
        src.row(0).colRange(0, win_center_distance+1).copyTo(
            padded_src.row(k).colRange(0, win_center_distance+1)
        );
        // copy to whats to the right of the row (diagonal)
        src.row(0).colRange(n_cols-win_center_distance-2, n_cols-1).copyTo(
            padded_src.row(k).colRange(win_center_distance+n_cols, win_center_distance*2+n_cols+1)
        );
    }
    for (int k = n_rows; k < n_rows+win_center_distance+1; k++)
    {
        // copy to central row
        src.row(n_rows-1).copyTo(
            padded_src.row(k).colRange(win_center_distance+1, win_center_distance+n_cols+1)
        );
        // copy to whats to the left of the row (diagonal)
        src.row(n_rows-1).colRange(0, win_center_distance+1).copyTo(
            padded_src.row(k).colRange(0, win_center_distance+1)
        );
        // copy to whats to the right of the row (diagonal)
        src.row(n_rows-1).colRange(n_cols-win_center_distance-2, n_cols-1).copyTo(
            padded_src.row(k).colRange(win_center_distance+n_cols, win_center_distance*2+n_cols+1)
        );
    }

    // fill src area
    src
        .rowRange(0, n_rows)
        .colRange(0, n_cols)
        .copyTo(
            padded_src
                .rowRange(win_center_distance+1, win_center_distance+n_rows+1)
                .colRange(win_center_distance+1, win_center_distance+n_cols+1)
        );

    // fill left and right borders
    for (int k = 0; k < win_center_distance+1; k++)
    {
        // copy to central row
        src.col(0).copyTo(
            padded_src.col(k).rowRange(win_center_distance+1, win_center_distance+n_rows+1)
        );
        // copy to whats to the left of the row (diagonal)
        src.col(0).rowRange(0, win_center_distance+1).copyTo(
            padded_src.col(k).rowRange(0, win_center_distance+1)
        );
        // copy to whats to the right of the row (diagonal)
        src.col(0).rowRange(n_rows-win_center_distance-2, n_rows-1).copyTo(
            padded_src.col(k).rowRange(win_center_distance+n_rows, win_center_distance*2+n_rows+1)
        );
    }
    for (int k = n_cols; k < n_cols+win_center_distance+1; k++)
    {
        // copy to central col
        src.col(n_cols-1).copyTo(
            padded_src.col(k).rowRange(win_center_distance+1, win_center_distance+n_rows+1)
        );
        // copy to whats to the left of the col (diagonal)
        src.col(n_cols-1).rowRange(0, win_center_distance+1).copyTo(
            padded_src.col(k).rowRange(0, win_center_distance+1)
        );
        // copy to whats to the right of the col (diagonal)
        src.col(n_cols-1).rowRange(n_rows-win_center_distance-2, n_rows-1).copyTo(
            padded_src.col(k).rowRange(win_center_distance+n_rows, win_center_distance*2+n_rows+1)
        );
    }

    // Equation 5 in the paper
    for (int y = 0; y < dark_channel.rows; y++)
    {
        for (int x = 0; x < dark_channel.cols; x++)
        {
            cv::Mat patch = padded_src
                            .rowRange(y, y+w_size+1)
                            .colRange(x, x+w_size+1);

            double min_red_channel = 0;
            double min_green_channel = 0;
            double min_blue_channel = 0;

            // Get the minimum value in the patch of each channel
            cv::Mat channels[3];
            cv::split(patch, channels);
            cv::minMaxLoc(channels[0], &min_blue_channel, &max_value);
            cv::minMaxLoc(channels[1], &min_green_channel, &max_value);
            cv::minMaxLoc(channels[2], &min_red_channel, &max_value);

            // Get lowest value among three channels
            min_value = (min_blue_channel > min_green_channel) ? min_green_channel : min_blue_channel;
            min_value = (min_value > min_red_channel) ? min_red_channel : min_value;
            min_value = min_green_channel;

            dark_channel.at<double>(y, x) = min_value;
        }
    }

    return dark_channel;
}

/**
 * @brief Estimate atmospheric light
 *
 * @param src       RGB image
 * @param w_size    Size of patch to consider (default is 15)
 *
 * @return Scalar   Estimated atmospheric light in the RGB channels
 */
static cv::Scalar estimate_atmospheric_light(const cv::Mat &input,
                                             const int w_size=15
) {
    std::vector<cv::Mat> channels;
    cv::split(input, channels);

    double min_pixel = 0;
    double AL_B = 0, AL_G = 0, AL_R = 0;

    cv::minMaxLoc(channels[0], &min_pixel, &AL_B);
    cv::minMaxLoc(channels[1], &min_pixel, &AL_G);
    cv::minMaxLoc(channels[2], &min_pixel, &AL_R);

    return cv::Scalar(AL_B, AL_G, AL_R);
}

/**
 * @brief Estimates the transmission map using the dark channel prior of the
 *        normalized image. A small fraction, omega, of the haze is kept to
 *        retain depth perspective after haze removal.
 *
 * @param src        3D Tensor in RGB format
 * @param omega      Fraction of haze to keep in image (default is 0.95)
 * @param w_size     Size of patch to consider (default is 15)
 *
 * @return cv::Mat   One-channel image describing the portion of the light that
 *                   is not scat-tered and reaches the camera.
 */
static cv::Mat estimate_transmission(const cv::Mat &src,
                                     const double omega=0.95,
                                     const int w_size=15
) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_64F);

    // Extract atmospheric light
    cv::Scalar A = estimate_atmospheric_light(src);

    // Normalize source image with atmospheric light
    cv::Mat norm_image (src.size(), CV_64FC3);
    // cv::divide(A, src, norm_image, CV_64FC3);
    for (int i = 0; i < norm_image.rows; i++)
    {
        for (int j = 0; j < norm_image.cols; j++)
        {
            cv::Vec3d orig = src.at<cv::Vec3d>(i, j);
            cv::Vec3d result;
                result[0] = orig[0] / A[0];
                result[1] = orig[1] / A[1];
                result[2] = orig[2] / A[2];
            norm_image.at<cv::Vec3d>(i, j) = result;
        }
    }

    // Get dark channel from normalized image
    cv::Mat norm_img_dc = get_dark_channel_prior(norm_image, w_size);

    // Compute final transmission
    dst = 1 - omega * norm_img_dc;

    return dst;
}

/**
 * @brief Similat to a soft matting algorithm to refine the transmission.
 *
 *        From http://kaiminghe.com/publications/eccv10guidedfilter.pdf
 *        and  https://arxiv.org/pdf/1505.00996.pdf
 *
 * @param src               Input RGB format
 * @param transmission      Computed transmission filter
 * @param omega             Window size (default is 60)
 * @param eps               Regularization parameter (default 0.01)
 *
 * @return cv::Mat          The refined alpha map
 */
static cv::Mat guided_filter(const cv::Mat &src,
                             const cv::Mat &transmission,
                             const int omega=60,
                             const double eps=0.01
) {
    // result
    cv::Mat f_alpha_map = cv::Mat::zeros(src.size(), CV_64FC1);

    // Sliding window size
    cv::Size w_size(omega, omega);

    // Split RGB channels
    cv::Mat channels[3];
    cv::split(src, channels);

    // get normalized value in each channel
    cv::Mat I_b(channels[0]);
    cv::Mat I_g(channels[1]);
    cv::Mat I_r(channels[2]);

    // Apply filters to normalized channels
    cv::Mat mean_I_r (I_r.size(), I_r.type());
    cv::Mat mean_I_g (I_g.size(), I_g.type());
    cv::Mat mean_I_b (I_b.size(), I_b.type());
    cv::blur(I_r, mean_I_r, w_size);
    cv::blur(I_g, mean_I_g, w_size);
    cv::blur(I_b, mean_I_b, w_size);

    // Apply filter to transmission (alpha) image
    cv::Mat mean_p(transmission.size(), transmission.type());
    cv::blur(transmission, mean_p, w_size);

    // Apply filters to image convoluted with transmission
    cv::Mat mean_Ip_r (I_r.size(), I_r.type());
    cv::Mat mean_Ip_g (I_g.size(), I_g.type());
    cv::Mat mean_Ip_b (I_b.size(), I_b.type());

    cv::blur(I_b.mul(transmission), mean_Ip_b, w_size);
    cv::blur(I_g.mul(transmission), mean_Ip_g, w_size);
    cv::blur(I_r.mul(transmission), mean_Ip_r, w_size);

    // Compute covariances
    cv::Mat cov_Ip_b(mean_Ip_b - mean_I_b.mul(mean_p));
    cv::Mat cov_Ip_g(mean_Ip_g - mean_I_g.mul(mean_p));
    cv::Mat cov_Ip_r(mean_Ip_r - mean_I_r.mul(mean_p));

    cv::Mat cov_Ip (src.size(), CV_64FC3);
    cv::Mat in[] = { cov_Ip_b, cov_Ip_g, cov_Ip_r  };
    int from_to[] = { 0,0, 1,1, 2,2 };
    cv::mixChannels( in, 3, &cov_Ip, 1, from_to, 3 );

    // Compute variances
    cv::Mat var_I_rr, var_I_rg, var_I_rb, var_I_gb, var_I_gg, var_I_bb;
    cv::blur(I_r.mul(I_r), var_I_rr, w_size);
    var_I_rr -= mean_I_r.mul(mean_I_r);

    cv::blur(I_r.mul(I_g), var_I_rg, w_size);
    var_I_rg -= mean_I_r.mul(mean_I_g);

    cv::blur(I_r.mul(I_b), var_I_rb, w_size);
    var_I_rb -= mean_I_r.mul(mean_I_b);

    cv::blur(I_g.mul(I_b), var_I_gb, w_size);
    var_I_gb -= mean_I_g.mul(mean_I_b);

    cv::blur(I_g.mul(I_g), var_I_gg, w_size);
    var_I_gg -= mean_I_g.mul(mean_I_g);

    cv::blur(I_b.mul(I_b), var_I_bb, w_size);
    var_I_bb -= mean_I_b.mul(mean_I_b);

    cv::Mat a(cv::Mat::zeros(src.size(), CV_64FC3));
    cv::Mat mat_e_id(eps * cv::Mat::eye(3, 3, CV_64F));
    cv::Mat Sigma(cv::Mat::zeros(cv::Size(3, 3), CV_64FC1));
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            // fill first row
            Sigma.at<double>(0, 0) = var_I_rb.at<double>(y, x);
            Sigma.at<double>(0, 1) = var_I_gb.at<double>(y, x);
            Sigma.at<double>(0, 2) = var_I_bb.at<double>(y, x);

            // fill second row
            Sigma.at<double>(1, 0) = var_I_rg.at<double>(y, x);
            Sigma.at<double>(1, 1) = var_I_gg.at<double>(y, x);
            Sigma.at<double>(1, 2) = var_I_gb.at<double>(y, x);

            // fill third row
            Sigma.at<double>(2, 0) = var_I_rr.at<double>(y, x);
            Sigma.at<double>(2, 1) = var_I_rg.at<double>(y, x);
            Sigma.at<double>(2, 2) = var_I_rb.at<double>(y, x);

            cv::Mat v(cov_Ip.at<cv::Vec3d>(y, x));
            cv::Mat inv((Sigma + mat_e_id).inv());
            cv::Vec3d result;
            result[0] = inv.at<cv::Vec3d>(0).dot(v);
            result[1] = inv.at<cv::Vec3d>(1).dot(v);
            result[2] = inv.at<cv::Vec3d>(2).dot(v);
            a.at<cv::Vec3d>(y, x) = result;
        }
    }

    std::vector<cv::Mat> blurred_channels;
    cv::split(a, blurred_channels);

    cv::blur(blurred_channels[0], blurred_channels[0], w_size);
    cv::blur(blurred_channels[1], blurred_channels[1], w_size);
    cv::blur(blurred_channels[2], blurred_channels[2], w_size);

    // Merge mean channels
    cv::Mat mean_a (src.size(), CV_64FC3);
    cv::Mat tmp3[] = { blurred_channels[0], blurred_channels[1], blurred_channels[2] };
    int from_to3[] = { 0,0, 1,1, 2,2 };
    cv::mixChannels( tmp3, 3, &mean_a, 1, from_to3, 3 );

    // Merge mean channels
    cv::Mat mean_I (src.size(), src.type());
    cv::Mat tmp[] = { mean_I_b, mean_I_g, mean_I_r };
    int from_to2[] = { 0,0, 1,1, 2,2 };
    cv::mixChannels( tmp, 3, &mean_I, 1, from_to2, 3 );

    cv::Mat s;
    cv::multiply(mean_I, a, s, 1, CV_64FC3);

    std::vector<cv::Mat> summed;
    cv::split(s, summed);
    cv::Mat b;
    cv::subtract(mean_p , (summed[0]+summed[1]+summed[2]), b, cv::Mat(), CV_64FC3);

    cv::Mat mean_b;
    cv::blur(b, mean_b, w_size);

    cv::Mat s2;
    cv::multiply(mean_a, src, s2, 1, CV_64FC3);

    summed.clear();
    cv::split(s2, summed);

    cv::add((summed[0]+summed[1]+summed[2]), mean_b, f_alpha_map, cv::Mat(), CV_64FC1);

    return f_alpha_map;
}

/**
 * @brief Implements the haze removal pipeline from Single Image Haze Removal
 *        Using Dark Channel Prior by He et al. (2009)
 *
 * @param original      Input RGB image
 * @param A             Reference to save the computed atmospheric light at
 * @param alpha_map     Reference to save the computed alpha map at
 * @param f_alpha_map   Reference to save the computed refined alpha map at
 * @param w_size        Window size of local patch (default is 15)
 * @param a_omega       Fraction of haze to keep in image (default is 0.95)
 * @param gf_w_size     Window size for guided filter (default is 200)
 * @param eps           Regularization parameter for guided filter(default 1e-6)
 *
 * @return cv::Mat          Image without haze
 */
static cv::Mat dehaze(const cv::Mat &original,
                      cv::Scalar &A,
                      cv::Mat &alpha_map,
                      cv::Mat &f_alpha_map,
                      const int w_size=15,
                      const double a_omega=0.95,
                      const int gf_w_size=200,
                      const double eps=1e-6
) {
    cv::Mat src;
    original.convertTo(src, CV_64FC3, (1.0/255.0));

    cv::Mat result = cv::Mat::zeros(src.size(), src.type());

    // atmospheric light A
    std::cout << "Computing atmospheric light" << std::endl;
    A = estimate_atmospheric_light(src);

    // get transmission diagram
    std::cout << "Computing transmission diagram" << std::endl;
    alpha_map = estimate_transmission(src, a_omega, w_size);


    // refine tansmission image
    std::cout << "Computing refine tansmission image" << std::endl;
    f_alpha_map = guided_filter(src, alpha_map, gf_w_size, eps);

    // get haze free image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double t_pixel = alpha_map.at<double>(i, j);
            cv::Vec3d haze_pixel = src.at<cv::Vec3d>(i, j);

            // "a small amount of haze are preserved in very dense haze region"
            double t = t_pixel > 0.1 ? t_pixel : 0.1;

            result.at<cv::Vec3d>(i, j)[0] = static_cast<double>(
                ((haze_pixel[0] - A[0]) / t) + A[0]);
            result.at<cv::Vec3d>(i, j)[1] = static_cast<double>(
                ((haze_pixel[1] - A[1]) / t) + A[1]);
            result.at<cv::Vec3d>(i, j)[2] = static_cast<double>(
                ((haze_pixel[2] - A[2]) / t) + A[2]);
        }
    }

    return result;
}
