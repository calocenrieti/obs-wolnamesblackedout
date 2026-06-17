#include "detect-filter-utils.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>

#include <opencv2/opencv.hpp>
using namespace cv;

std::string trim_copy(const std::string &s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(start, end - start);
}

std::vector<std::string> split_comma_list(const std::string &s)
{
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string token = trim_copy(s.substr(i, j - i));
        if (!token.empty()) out.push_back(token);
        i = j + 1;
    }
    return out;
}

std::string sanitize_ocr_text(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == '\'' || c == ' ') {
            out.push_back(c);
        }
    }
    return out;
}

int levenshtein_distance(const std::string &a, const std::string &b)
{
    const size_t n = a.size();
    const size_t m = b.size();
    if (n == 0) return static_cast<int>(m);
    if (m == 0) return static_cast<int>(n);

    std::vector<int> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= m; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost });
        }
        prev.swap(cur);
    }
    return prev[m];
}

double levenshtein_similarity(const std::string &a_in, const std::string &b_in)
{
    std::string a = a_in;
    std::string b = b_in;
    if (a.empty() && b.empty()) return 1.0;

    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    int dist = levenshtein_distance(a, b);
    int maxlen = std::max(static_cast<int>(a.size()), static_cast<int>(b.size()));
    if (maxlen == 0) return 1.0;
    return 1.0 - static_cast<double>(dist) / static_cast<double>(maxlen);
}

bool is_partial_match(const std::string &ocr_text, const std::string &exclude_text)
{
    size_t ocr_space = ocr_text.find(' ');
    size_t exclude_space = exclude_text.find(' ');

    {
        std::string ocr_first = (ocr_space != std::string::npos) ? ocr_text.substr(0, ocr_space) : ocr_text;
        std::string exclude_first = (exclude_space != std::string::npos) ? exclude_text.substr(0, exclude_space) : exclude_text;

        if (!ocr_first.empty() && !exclude_first.empty()) {
            double ratio = static_cast<double>(ocr_first.length()) / static_cast<double>(exclude_first.length());
            if (ratio >= 0.7 && exclude_first.find(ocr_first) != std::string::npos) {
                return true;
            }
        }
    }

    {
        std::string ocr_second = (ocr_space != std::string::npos) ? ocr_text.substr(ocr_space + 1) : "";
        std::string exclude_second = (exclude_space != std::string::npos) ? exclude_text.substr(exclude_space + 1) : "";

        if (!ocr_second.empty() && !exclude_second.empty()) {
            double ratio = static_cast<double>(ocr_second.length()) / static_cast<double>(exclude_second.length());
            if (ratio >= 0.7 && exclude_second.find(ocr_second) != std::string::npos) {
                return true;
            }
        }
    }

    return false;
}

void drawDashedLine(Mat &img, Point pt1, Point pt2, Scalar color, int thickness, int lineType,
		    int dashLength)
{
	double lineLength = norm(pt1 - pt2);
	double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);

	Point p1 = pt1;
	Point p2;
	bool draw = true;

	for (double d = 0; d < lineLength; d += dashLength) {
		if (draw) {
			p2.x = pt1.x +
			       static_cast<int>(cos(angle) * std::min(d + dashLength, lineLength));
			p2.y = pt1.y +
			       static_cast<int>(sin(angle) * std::min(d + dashLength, lineLength));
			line(img, p1, p2, color, thickness, lineType);
		}
		p1.x = pt1.x + static_cast<int>(cos(angle) * (d + dashLength));
		p1.y = pt1.y + static_cast<int>(sin(angle) * (d + dashLength));
		draw = !draw;
	}
}

void drawDashedRectangle(Mat &img, Rect rect, Scalar color, int thickness, int lineType,
			 int dashLength)
{
	Point pt1(rect.x, rect.y);
	Point pt2(rect.x + rect.width, rect.y);
	Point pt3(rect.x + rect.width, rect.y + rect.height);
	Point pt4(rect.x, rect.y + rect.height);

	drawDashedLine(img, pt1, pt2, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt2, pt3, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt3, pt4, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt4, pt1, color, thickness, lineType, dashLength);
}
