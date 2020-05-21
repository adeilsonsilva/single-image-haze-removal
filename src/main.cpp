/**
 * @file main.cpp
 * @author Adeilson Silva (adeilsonsilva@dcc.ufba.br)
 * @brief It implements haze removal on an user-defined image and displays it with OpenCV.
 * @version 1.0
 * @date 2020-05-21
 *
 * @copyright Copyright (c) 2020
 *
 */

#include <haze_removal.hpp>

int main(int argc, char const *argv[])
{

    if (argc != 2) {
        std::cout << "Usage: ./haze_removal <DATASET_PATH>" << std::endl;
        return -1;
    }

    std::cout << "Press ESC to exit." << std::endl;

    std::string path = argv[1];
    std::cout << "Loading image: " << path << std::endl;

    cv::Mat original = cv::imread(path);            // input
    cv::Mat alpha_map(original.size(), CV_64F);     // output
    cv::Mat f_alpha_map(original.size(), CV_64F);   // output
    cv::Scalar atmospheric_light;                   // output

    std::cout << "Computing haze free image" << std::endl;
    cv::Mat result = dehaze(
        original,
        atmospheric_light,
        alpha_map,
        f_alpha_map
    );

    cv::namedWindow(W_ORIGINAL, CV_WINDOW_AUTOSIZE);
    cv::imshow(W_ORIGINAL, original);

    cv::namedWindow(W_TRANSMISSION_IMAGE, CV_WINDOW_AUTOSIZE);
    cv::imshow(W_TRANSMISSION_IMAGE, alpha_map);

    cv::namedWindow(W_REFINED_TRANSMISSION_IMAGE, CV_WINDOW_AUTOSIZE);
    cv::imshow(W_REFINED_TRANSMISSION_IMAGE, f_alpha_map);

    cv::namedWindow(W_RESULT, CV_WINDOW_AUTOSIZE);
    cv::imshow(W_RESULT, result);

    while (cv::waitKey(10000) != 27);

    return 0;
}

