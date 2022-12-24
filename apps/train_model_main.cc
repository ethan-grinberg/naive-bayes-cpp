#include <iostream>

#include <core/trainer.h>
#include <core/classifier.h>

using naivebayes::Trainer;
using naivebayes::Classifier;

int main() {

  //Inputs training images to the trainer and trains model.
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-"
      "ethanbg2\\data\\trainingimagesandlabels.txt";

  Trainer trainer(28, 10,1);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> trainer;
  } else {
    std::cout << "file not found";
  }

  std::cout << trainer.GetImageCount();

  //Outputs the trained model to a file.
  std::string output_file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\model.txt";
  std::ofstream output_file(output_file_name);
  output_file << trainer;

  return 0;
}
