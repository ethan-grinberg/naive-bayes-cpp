#include <core/trainer.h>

#include <iostream>
#include <sstream>

namespace naivebayes {

Trainer::Trainer(int image_size, int num_classes, int k_value) {
  image_size_ = image_size;
  image_count_ = 0;
  num_classes_ = num_classes;
  k_value_ = k_value;

  for (int i = 0; i < num_classes; i++) {
    shaded_pixels_.insert(std::make_pair(i, Image(image_size_)));
    unshaded_pixels_.insert(std::make_pair(i, Image(image_size_)));
  }

  vector<vector<vector<vector<double>>>> vect(
      image_size_, vector<vector<vector<double>>>(
                       image_size_, vector<vector<double>>(
                                        num_classes_, vector<double>(kNumShades_, 0))));

  feature_probs_ = vect;
}

std::istream &operator>>(std::istream &is, Trainer& trainer) {
    size_t line_number = 0;
    int image_class;
    int image_row;

    std::string line;
    while (getline(is, line)) {
      //This means the file reader is on a new image and this line is an
      // image label.
      if (line_number % (trainer.image_size_ + 1) == 0) {
        //The current image class is updated.
        std::stringstream num(line);
        num >> image_class;

        //Resets the image_row to be zero because there's a new image.
        image_row = 0;

        trainer.image_count_++;
        line_number++;
        continue;
      }

      trainer.ProcessLine(line, image_class, image_row);
      image_row++;

      line_number++;
    }

    // Initializes the priors and feature probabilities based on the training
    // images.
    trainer.InitializePriors();
    trainer.InitializeFeatureProbs();
    return is;
}

std::ostream& operator<<(std::ostream& os, Trainer& trainer) {

  for (int i = 0; i < trainer.num_classes_; i++) {
    os << trainer.priors_[i] << std::endl;
    for (int j = 0; j < trainer.image_size_; j++) {

      for (int k = 0; k < trainer.image_size_; k++) {

        if (k == (trainer.image_size_ - 1)) {
          os << trainer.feature_probs_[j][k][i][0];
          break;
        }

        os << trainer.feature_probs_[j][k][i][0] << " ";
      }

      os << std::endl;
    }
  }
  return os;
}

void Trainer::ProcessLine(std::string& line, int image_class, int row_num) {
  for (size_t char_index = 0; char_index < line.length(); char_index++) {
    char current_char = line.at(char_index);

    if (current_char == '#' || current_char == '+') {
      shaded_pixels_.at(image_class).AppendPixelCount(row_num, char_index);
    } else {
      unshaded_pixels_.at(image_class).AppendPixelCount(row_num, char_index);
    }
  }
}

double Trainer::CalculatePrior(int image_class) {
  return (double)(k_value_ + (unshaded_pixels_.at(image_class).GetPixel(0, 0) +
                              shaded_pixels_.at(image_class).GetPixel(0, 0))) /
         ((num_classes_ * k_value_) + image_count_);
}

void Trainer::InitializePriors() {
  for (int i = 0; i < num_classes_; i++) {
    priors_.push_back(CalculatePrior(i));
  }
}

map<int, Image> Trainer::GetUnshadedPixels() const {
  return unshaded_pixels_;
}

map<int, Image> Trainer::GetShadedPixels() const {
  return shaded_pixels_;
}

int Trainer::GetImageCount() const {
  return image_count_;
}

vector<double> Trainer::GetPriors() const {
  return priors_;
}

double Trainer::CalculateFeatureProb(int image_class, bool shaded, int row_num,
                                     int col_num) {
  if (shaded) {
    return (double) (k_value_ +
                    (shaded_pixels_.at(image_class).GetPixel(row_num, col_num))) /
           ((kNumShades_ * k_value_) +
            (unshaded_pixels_.at(image_class).GetPixel(row_num, col_num) +
             shaded_pixels_.at(image_class).GetPixel(row_num, col_num)));
  } else {
    return (double) (k_value_ +
                     (unshaded_pixels_.at(image_class).GetPixel(row_num, col_num))) /
        ((kNumShades_ * k_value_) +
         (unshaded_pixels_.at(image_class).GetPixel(row_num, col_num) +
             shaded_pixels_.at(image_class).GetPixel(row_num, col_num)));
  }
}

void Trainer::InitializeFeatureProbs() {
  for (size_t i = 0; i < feature_probs_.size(); i++) {
    for (size_t j = 0; j < feature_probs_[i].size(); j++) {
      for (size_t k = 0; k < feature_probs_[i][j].size(); k++) {
        feature_probs_[i][j][k][0] = CalculateFeatureProb(k, false, i, j);
        feature_probs_[i][j][k][1] = CalculateFeatureProb(k, true, i, j);
      }
    }
  }
}

vector<vector<vector<vector<double>>>> Trainer::GetFeatureProbs() const {
  return feature_probs_;
}

}  // namespace naivebayes