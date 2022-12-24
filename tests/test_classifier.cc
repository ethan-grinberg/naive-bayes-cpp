#include <core/classifier.h>

#include <catch2/catch.hpp>
#include <iostream>

using naivebayes::Classifier;

TEST_CASE("Read in model") {
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_model.txt";

  Classifier classifier(3, 2);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> classifier;
  } else {
    std::cout << "file not found";
  }

  SECTION("Unshaded class 0") {
    REQUIRE(classifier.GetFeatureProbs()[0][0][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[0][1][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[0][2][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[1][0][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[1][1][0][0] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[1][2][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[2][0][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[2][1][0][0] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[2][2][0][0] == Approx(.333333333));
  }

  SECTION("Shaded class 0") {
    REQUIRE(classifier.GetFeatureProbs()[0][0][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[0][1][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[0][2][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[1][0][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[1][1][0][1] == Approx(.333333333));
    REQUIRE(classifier.GetFeatureProbs()[1][2][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[2][0][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[2][1][0][1] == Approx(.666666667));
    REQUIRE(classifier.GetFeatureProbs()[2][2][0][1] == Approx(.666666667));
  }

  SECTION("Unshaded class 1") {
    REQUIRE(classifier.GetFeatureProbs()[0][0][1][0] == Approx(.5));
    REQUIRE(classifier.GetFeatureProbs()[0][1][1][0] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[0][2][1][0] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[1][0][1][0] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[1][1][1][0] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[1][2][1][0] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[2][0][1][0] == Approx(.5));
    REQUIRE(classifier.GetFeatureProbs()[2][1][1][0] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[2][2][1][0] == Approx(.5));
  }

  SECTION("Shaded class 1") {
    REQUIRE(classifier.GetFeatureProbs()[0][0][1][1] == Approx(.5));
    REQUIRE(classifier.GetFeatureProbs()[0][1][1][1] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[0][2][1][1] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[1][0][1][1] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[1][1][1][1] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[1][2][1][1] == Approx(.25));
    REQUIRE(classifier.GetFeatureProbs()[2][0][1][1] == Approx(.5));
    REQUIRE(classifier.GetFeatureProbs()[2][1][1][1] == Approx(.75));
    REQUIRE(classifier.GetFeatureProbs()[2][2][1][1] == Approx(.5));
  }

  SECTION("Test Priors") {
    REQUIRE(classifier.GetPriors().at(0) == Approx(.4));
    REQUIRE(classifier.GetPriors().at(1) == Approx(.6));
  }
}

TEST_CASE("Classification") {
  std::string file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_model.txt";

  Classifier classifier(3, 2);

  std::ifstream input_file(file_name);

  if (input_file.is_open()) {
    input_file >> classifier;
  } else {
    std::cout << "file not found";
  }

  std::string test_input_file =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\test_input.txt";

  std::ifstream test_input(test_input_file);

  classifier.ReadInputImage(test_input);

  REQUIRE(classifier.GetHighestLikelihood() == 1);

  SECTION("Likelihood score calculations") {
    SECTION("Likelihood of it being 1") {
      REQUIRE(classifier.GetLikelihoodScores().at(1) == Approx(-1.874571156));
    }
    SECTION("Likelihood of it being 0") {
      REQUIRE(classifier.GetLikelihoodScores().at(0) == Approx(-3.788941314));
    }
  }
}

TEST_CASE("Classifier accuracy, sanity check, (28x28 images)") {
  //Read in model to Classifier
  Classifier classifier(28, 10);
  std::string model_file_name =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\model.txt";

  std::ifstream model(model_file_name);

  std::string test_classifier_file =
      "C:\\Users\\ethan\\Documents\\School\\CS126\\Cinder\\my-projects\\naive-"
      "bayes-ethanbg2\\data\\testimagesandlabels.txt";

  model >> classifier;

  std::ifstream test_classifier(test_classifier_file);

  REQUIRE(classifier.ValidateClassifier(test_classifier) >= 0.7);
}

