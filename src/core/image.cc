#include "core/image.h"

namespace naivebayes {

Image::Image(int image_size) {
  std::vector<vector<int>> pixels(image_size, vector<int>(image_size));
  pixels_ = pixels;
  image_size_ = image_size;
}


void Image::AppendPixelCount(int row, int column) {
  pixels_[row][column] = pixels_[row][column] + 1;
}

int Image::GetPixel(int row_num, int column_num) const {
  return pixels_[row_num][column_num];
}

void Image::ShadePixel(int row, int col) {
  pixels_[row][col] = 1;
}

void Image::ClearPixels() {
  std::vector<vector<int>> pixels(image_size_, vector<int>(image_size_));
  pixels_ = pixels;
}

} // namespace naivebayes
