#pragma once

#include <fstream>
#include <vector>

namespace naivebayes {

using std::vector;

/**
 * This class reads in a trained model from a file
 * and classifies an input image.
 */

class Classifier{
 public:
  /**
   * The image size the classifier takes in must be the same image size
   * as the training images. It also must have the same number of possible
   * classes as the training images.
   */
  Classifier(int image_size, int num_classes);

  Classifier() {}

  /**
   * Reads in a trained model from a file. Stores all the feature and
   * prior probabilities from the file.
   */
  friend std::istream &operator>>(std::istream &is, Classifier& classifier);

  /**
   * Reads a new image for classification, this initializes the likelihood
   * scores for classification.
   */
  std::istream& ReadInputImage(std::istream &is);

  /**
   * Reads in images with a label above each from a file to test the accuracy
   * of the classifier. It returns an accuracy score based on the files images.
   */
  double ValidateClassifier(std::istream &is);

  vector<vector<vector<vector<double>>>> GetFeatureProbs() const;

  vector<double> GetPriors() const;

  int GetHighestLikelihood() const;

  vector<double> GetLikelihoodScores() const;

 private:
  //{class 0, class1, class2... class9}
  vector<double> priors_;

  //Stored in the format feature_probs_[row][col][class][shade]
  vector<vector<vector<vector<double>>>> feature_probs_;

  //{class 0, class1, class2... class9}
  vector<double> likelihood_scores_;

  int image_size_;

  int num_classes_;

  /**
   * Processes a line from an image. Uses each pixel from the image to update
   * the likelihood score of each class.
   */
  void ProcessImageLine(std::string& line, int row_num);

  /**
   * Resets the likelihood scores and initializes the likelihood score of each
   * class to be the log of the corresponding class prior probability.
   */
  void InitializeLikelihood();

  /**
   * Based on an image's pixel, the likelihood of each class is added on to.
   * It takes the log of the corresponding feature probability.
   */
  void AddToLikelihood(int row, int col, int shade);
};

} // namespace naivebayes
