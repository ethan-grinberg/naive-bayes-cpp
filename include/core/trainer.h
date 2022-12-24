#pragma once

#include <map>
#include <string>
#include <vector>
#include <fstream>


#include "image.h"

using std::map;
using std::vector;

namespace naivebayes {

/**
 * This class reads in a training images, calculates feature and prior
 * probabilities and outputs them to a file.
 */
class Trainer {
 public:

  /**
   * Creates a trainer object with specified k_value for LaPlace Smoothing,
   * training images size, and the number of possible classes.
   */
  Trainer(int image_size, int num_classes, int k_value);

  map<int, Image> GetUnshadedPixels() const;

  map<int, Image> GetShadedPixels() const;

  int GetImageCount() const;

  vector<double> GetPriors() const;

  vector<vector<vector<vector<double>>>> GetFeatureProbs() const;

  /**
   * Reads in a training images to a file and uses them to calculate feature
   * and prior probabilities.
   */
  friend std::istream &operator>>(std::istream &is, Trainer& trainer);

  /**
   * Outputs the prior probability for each class and the unshaded
   * feature probabilities for each class. The first line of each image class
   * is the prior probability and the feature probabilities for each pixel are
   * arranged in the same format as an image's pixels. Shaded feature
   * probabilities are excluded to save time reading in the
   * model to a classifier.
   */
  friend std::ostream &operator<<(std::ostream &os, Trainer& trainer);

 private:
  //Count of each unshaded pixel for each image class.
  map <int, Image> unshaded_pixels_;

  //Count of each shaded pixel for each image class.
  map <int, Image> shaded_pixels_;

  int image_size_;

  int image_count_;

  int num_classes_;

  int k_value_;

  static const int kNumShades_ = 2;

  vector<double> priors_;

  //Stored in the format feature_probs_[row][col][class][shade]
  vector<vector<vector<vector<double>>>> feature_probs_;

  /**
   * Helper function to read each line of an image and properly add to the
   * count of shaded and unshaded pixels of each class.
   */
  void ProcessLine(std::string& line, int image_class, int row_num);

  /**
   * Helper function to store each calculated prior for each class into a vector.
   */
  void InitializePriors();

  /**
   * Helper function to store each calculated feature probability into a vector.
   */
  void InitializeFeatureProbs();

  /**
   * Calculates a prior probability for a specific image class.
   */
  double CalculatePrior(int image_class);

  /**
   * Calculates a feature probability based on a specific image class, if it's
   * shaded or not, and specific pixel.
   */
  double CalculateFeatureProb(int image_class, bool shaded, int row_num,
                              int col_num);
};

}  // namespace naivebayes
