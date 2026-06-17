#ifndef DETECT_FILTER_UTILS_H
#define DETECT_FILTER_UTILS_H

#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

std::string trim_copy(const std::string &s);
std::vector<std::string> split_comma_list(const std::string &s);
std::string sanitize_ocr_text(const std::string &text);
int levenshtein_distance(const std::string &a, const std::string &b);
double levenshtein_similarity(const std::string &a_in, const std::string &b_in);
bool is_partial_match(const std::string &ocr_text, const std::string &exclude_text);

void drawDashedLine(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness = 1,
		    int lineType = 8, int dashLength = 10);

void drawDashedRectangle(cv::Mat &img, cv::Rect rect, cv::Scalar color, int thickness = 1,
			 int lineType = 8, int dashLength = 10);

#endif // DETECT_FILTER_UTILS_H
