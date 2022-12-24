#pragma once

#include <vector>

namespace naivebayes {

using std::vector;

/**
 * Class representing an abstract image.
 */
class Image {
 public:
  /**
   * Initializes a 2D vector to zeros based on the image size.
   */
  Image(int image_size);

  /**
  * Gets the specified pixel from the image.
  */
  int GetPixel(int row_num, int column_num) const;

  /**
  * The corresponding row and column is set to 1.
  */
  void ShadePixel(int row, int col);

  /**
  * Each pixel's value is set to 0.
  */
  void ClearPixels();

  /**
  * Adds one to the specified pixel
  */
  void AppendPixelCount(int row_num, int column_num);

 private:
  vector<vector<int>> pixels_;

  int image_size_;
};

} // namespace naivebayes
