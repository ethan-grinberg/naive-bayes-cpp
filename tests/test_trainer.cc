#include <core/trainer.h>

#include <catch2/catch.hpp>
#include <iostream>

using naivebayes::Trainer;

TEST_CASE("Read training images") {
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_training_data.txt";

  Trainer trainer(3, 2,1);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> trainer;
  } else {
    std::cout << "file not found";
  }
  SECTION("test shaded pixel count") {
    naivebayes::Image shaded_0 = trainer.GetShadedPixels().at(0);
    REQUIRE(shaded_0.GetPixel(0, 0) == 1);
    REQUIRE(shaded_0.GetPixel(0, 1) == 1);
    REQUIRE(shaded_0.GetPixel(0, 2) == 1);
    REQUIRE(shaded_0.GetPixel(1, 0) == 1);
    REQUIRE(shaded_0.GetPixel(1, 1) == 0);
    REQUIRE(shaded_0.GetPixel(1, 2) == 1);
    REQUIRE(shaded_0.GetPixel(2, 0) == 1);
    REQUIRE(shaded_0.GetPixel(2, 1) == 1);
    REQUIRE(shaded_0.GetPixel(2, 2) == 1);

    naivebayes::Image shaded_1 = trainer.GetShadedPixels().at(1);

    REQUIRE(shaded_1.GetPixel(0, 0) == 1);
    REQUIRE(shaded_1.GetPixel(0, 1) == 2);
    REQUIRE(shaded_1.GetPixel(0, 2) == 0);
    REQUIRE(shaded_1.GetPixel(1, 0) == 0);
    REQUIRE(shaded_1.GetPixel(1, 1) == 2);
    REQUIRE(shaded_1.GetPixel(1, 2) == 0);
    REQUIRE(shaded_1.GetPixel(2, 0) == 1);
    REQUIRE(shaded_1.GetPixel(2, 1) == 2);
    REQUIRE(shaded_1.GetPixel(2, 2) == 1);
  }
  SECTION("test unshaded pixel count") {
    naivebayes::Image unshaded_0 = trainer.GetUnshadedPixels().at(0);

    REQUIRE(unshaded_0.GetPixel(0, 0) == 0);
    REQUIRE(unshaded_0.GetPixel(0, 1) == 0);
    REQUIRE(unshaded_0.GetPixel(0, 2) == 0);
    REQUIRE(unshaded_0.GetPixel(1, 0) == 0);
    REQUIRE(unshaded_0.GetPixel(1, 1) == 1);
    REQUIRE(unshaded_0.GetPixel(1, 2) == 0);
    REQUIRE(unshaded_0.GetPixel(2, 0) == 0);
    REQUIRE(unshaded_0.GetPixel(2, 1) == 0);
    REQUIRE(unshaded_0.GetPixel(2, 2) == 0);

    naivebayes::Image unshaded_1 = trainer.GetUnshadedPixels().at(1);

    REQUIRE(unshaded_1.GetPixel(0, 0) == 1);
    REQUIRE(unshaded_1.GetPixel(0, 1) == 0);
    REQUIRE(unshaded_1.GetPixel(0, 2) == 2);
    REQUIRE(unshaded_1.GetPixel(1, 0) == 2);
    REQUIRE(unshaded_1.GetPixel(1, 1) == 0);
    REQUIRE(unshaded_1.GetPixel(1, 2) == 2);
    REQUIRE(unshaded_1.GetPixel(2, 0) == 1);
    REQUIRE(unshaded_1.GetPixel(2, 1) == 0);
    REQUIRE(unshaded_1.GetPixel(2, 2) == 1);
  }

  SECTION("test image count") {
    REQUIRE(trainer.GetImageCount() == 3);
  }
}

TEST_CASE("Read training images of different sizes (6x6)") {
  std::string test_file =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_training_data_dif_size.txt";

  std::ifstream input_file(test_file);

  Trainer trainer(6, 2,1);

  input_file >> trainer;

  naivebayes::Image shaded_1 = trainer.GetShadedPixels().at(1);

  REQUIRE(shaded_1.GetPixel(0, 0) == 1);
  REQUIRE(shaded_1.GetPixel(0, 1) == 1);
  REQUIRE(shaded_1.GetPixel(0, 2) == 1);
  REQUIRE(shaded_1.GetPixel(0, 3) == 0);
  REQUIRE(shaded_1.GetPixel(0, 4) == 0);
  REQUIRE(shaded_1.GetPixel(0, 5) == 0);

  REQUIRE(shaded_1.GetPixel(1, 0) == 0);
  REQUIRE(shaded_1.GetPixel(1, 1) == 0);
  REQUIRE(shaded_1.GetPixel(1, 2) == 1);
  REQUIRE(shaded_1.GetPixel(1, 3) == 0);
  REQUIRE(shaded_1.GetPixel(1, 4) == 0);
  REQUIRE(shaded_1.GetPixel(1, 5) == 0);


  REQUIRE(shaded_1.GetPixel(2, 0) == 0);
  REQUIRE(shaded_1.GetPixel(2, 1) == 0);
  REQUIRE(shaded_1.GetPixel(2, 2) == 1);
  REQUIRE(shaded_1.GetPixel(2, 3) == 0);
  REQUIRE(shaded_1.GetPixel(2, 4) == 0);
  REQUIRE(shaded_1.GetPixel(2, 5) == 0);

  REQUIRE(shaded_1.GetPixel(3, 0) == 0);
  REQUIRE(shaded_1.GetPixel(3, 1) == 0);
  REQUIRE(shaded_1.GetPixel(3, 2) == 1);
  REQUIRE(shaded_1.GetPixel(3, 3) == 0);
  REQUIRE(shaded_1.GetPixel(3, 4) == 0);
  REQUIRE(shaded_1.GetPixel(3, 5) == 0);

  REQUIRE(shaded_1.GetPixel(4, 0) == 0);
  REQUIRE(shaded_1.GetPixel(4, 1) == 0);
  REQUIRE(shaded_1.GetPixel(4, 2) == 1);
  REQUIRE(shaded_1.GetPixel(4, 3) == 0);
  REQUIRE(shaded_1.GetPixel(4, 4) == 0);
  REQUIRE(shaded_1.GetPixel(4, 5) == 0);

  REQUIRE(shaded_1.GetPixel(5, 0) == 1);
  REQUIRE(shaded_1.GetPixel(5, 1) == 1);
  REQUIRE(shaded_1.GetPixel(5, 2) == 1);
  REQUIRE(shaded_1.GetPixel(5, 3) == 1);
  REQUIRE(shaded_1.GetPixel(5, 4) == 1);
  REQUIRE(shaded_1.GetPixel(5, 5) == 1);
}

TEST_CASE("Test priors") {
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_training_data.txt";

  Trainer trainer(3, 2,1);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> trainer;
  } else {
    std::cout << "file not found";
  }

  REQUIRE(trainer.GetPriors().at(0) == Approx(.4));
  REQUIRE(trainer.GetPriors().at(1) == Approx(.6));
}

TEST_CASE("Feature Probability Calculations") {
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_training_data.txt";

  Trainer trainer(3, 2,1);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> trainer;
  } else {
    std::cout << "file not found";
  }

  SECTION("Unshaded class 0") {
    REQUIRE(trainer.GetFeatureProbs()[0][0][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[0][1][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[0][2][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[1][0][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[1][1][0][0] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[1][2][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[2][0][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[2][1][0][0] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[2][2][0][0] == Approx(.333333333));
  }

  SECTION("Shaded class 0") {
    REQUIRE(trainer.GetFeatureProbs()[0][0][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[0][1][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[0][2][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[1][0][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[1][1][0][1] == Approx(.333333333));
    REQUIRE(trainer.GetFeatureProbs()[1][2][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[2][0][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[2][1][0][1] == Approx(.666666667));
    REQUIRE(trainer.GetFeatureProbs()[2][2][0][1] == Approx(.666666667));
  }

  SECTION("Unshaded class 1") {
    REQUIRE(trainer.GetFeatureProbs()[0][0][1][0] == Approx(.5));
    REQUIRE(trainer.GetFeatureProbs()[0][1][1][0] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[0][2][1][0] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[1][0][1][0] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[1][1][1][0] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[1][2][1][0] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[2][0][1][0] == Approx(.5));
    REQUIRE(trainer.GetFeatureProbs()[2][1][1][0] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[2][2][1][0] == Approx(.5));
  }

  SECTION("Shaded class 1") {
    REQUIRE(trainer.GetFeatureProbs()[0][0][1][1] == Approx(.5));
    REQUIRE(trainer.GetFeatureProbs()[0][1][1][1] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[0][2][1][1] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[1][0][1][1] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[1][1][1][1] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[1][2][1][1] == Approx(.25));
    REQUIRE(trainer.GetFeatureProbs()[2][0][1][1] == Approx(.5));
    REQUIRE(trainer.GetFeatureProbs()[2][1][1][1] == Approx(.75));
    REQUIRE(trainer.GetFeatureProbs()[2][2][1][1] == Approx(.5));
  }
}

TEST_CASE("Writing to a file") {
  Trainer trainer(3, 2,1);

  std::string input_file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_training_data.txt";

  std::ifstream input_file(input_file_name);
  if (input_file.is_open()) {
    input_file >> trainer;
  } else {
    std::cout << "file not found";
  }

  std::string output_file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_model.txt";
  std::ofstream output_file(output_file_name);
  output_file << trainer;

}

