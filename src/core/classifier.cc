#include <core/classifier.h>

#include <iostream>
#include <sstream>

namespace naivebayes {

Classifier::Classifier(int image_size, int num_classes) {
  image_size_ = image_size;
  num_classes_ = num_classes;

  vector<vector<vector<vector<double>>>> vect(
      image_size_, vector<vector<vector<double>>>(
          image_size_, vector<vector<double>>(
              num_classes_, vector<double>(2, 0))));

  feature_probs_ = vect;
}

std::istream& operator>>(std::istream& is, Classifier& classifier) {
  size_t line_number = 0;

  //Starts at -1 because first image class is 0 and it increments at the first
  // image class.
  int class_num = -1;
  int image_row;

  std::string line;
  while (getline(is, line)) {
    std::stringstream line_stream(line);

    //This means the file reader is on a new image class's prior probability
    if (line_number % (classifier.image_size_ + 1) == 0) {
      double prior;
      line_stream >> prior;

      classifier.priors_.push_back(prior);

      class_num++;
      line_number++;
      image_row = 0;
      continue;
    }

    std::string value;
    //Resets the column number to 0
    int column_num = 0;
    //Reads in the feature probabilities that are separated by spaces.
    while (line_stream >> value) {
      double prob = stod(value);

      classifier.feature_probs_[image_row][column_num][class_num][0] = prob;
      //model file only has unshaded probabilities as to not waste time reading
      // extra lines. Shaded probabilities can easily be calculated.
      classifier.feature_probs_[image_row][column_num][class_num][1] = (1 - prob);

      column_num++;
    }

    line_number++;
    image_row++;
  }
  return is;
}

vector<vector<vector<vector<double>>>> Classifier::GetFeatureProbs() const {
  return feature_probs_;
}

vector<double> Classifier::GetPriors() const {
  return priors_;
}

std::istream& Classifier::ReadInputImage(std::istream &is) {
  InitializeLikelihood();

  size_t line_number = 0;

  std::string line;
  while (getline(is, line)) {
    ProcessImageLine(line, line_number);
    line_number++;
  }
  return is;
}

void Classifier::ProcessImageLine(std::string& line, int row_num) {
  for (size_t char_index = 0; char_index < line.length(); char_index++) {
    char current_char = line.at(char_index);

    if (current_char == '#' || current_char == '+') {
      AddToLikelihood(row_num, char_index, 1);
    } else {
      AddToLikelihood(row_num, char_index, 0);
    }
  }
}

void Classifier::InitializeLikelihood() {
  likelihood_scores_.clear();
  for (size_t i = 0; i < priors_.size(); i++) {
    //Made negative so likelihoods are positive
    likelihood_scores_.push_back(log10(priors_[i]));
  }
}

void Classifier::AddToLikelihood(int row, int col, int shade) {
  for (size_t i = 0; i < likelihood_scores_.size(); i++) {
    //Made negative so likelihoods are positive
    likelihood_scores_[i] += log10(feature_probs_[row][col][i][shade]);
  }
}

int Classifier::GetHighestLikelihood() const{
  double max = likelihood_scores_[0];
  int best_class = 0;

  for (size_t num_class = 0; num_class < likelihood_scores_.size(); num_class++) {
    if (likelihood_scores_[num_class] > max) {
      max = likelihood_scores_[num_class];
      best_class = num_class;
    }
  }

  return best_class;
}

double Classifier::ValidateClassifier(std::istream& is) {
  int num_correct = 0;
  int num_images = 0;

  size_t line_number = 0;
  int image_class;
  int image_row;

  std::string line;
  while (getline(is, line)) {
    //This means the file reader is on a new image and this line is an
    // image label.
    if (line_number % (image_size_ + 1) == 0) {
      std::stringstream num(line);
      num >> image_class;

      image_row = 0;

      num_images++;
      line_number++;
      //Resets the likelihoods because a new image is about to be classified.
      InitializeLikelihood();
      continue;
    }


    ProcessImageLine(line, image_row);
    image_row++;

    //Classifies image once it gets to the end of the image.
    if (image_row == image_size_) {
      int prediction = GetHighestLikelihood();
      if (prediction == image_class) {
        num_correct++;
      }
    }

    line_number++;
  }

  return static_cast<double>((double) num_correct / (double) num_images);
}

vector<double> Classifier::GetLikelihoodScores() const {
  return likelihood_scores_;
}

} // namespace naivebayes

