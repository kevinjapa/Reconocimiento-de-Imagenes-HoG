// #include <opencv2/opencv.hpp>
// #include <opencv2/ml.hpp>
// #include <iostream>
// #include <vector>
// #include <filesystem>
// #include <map>
// #include <fstream>

// using namespace cv;
// using namespace cv::ml;
// using namespace std;
// namespace fs = std::filesystem;

// // Función para cargar imágenes y etiquetas
// void loadImagesAndLabels(const string& directory, vector<Mat>& images, vector<int>& labels, map<string, int>& labelMap, map<int, string>& reverseLabelMap) {
//     int currentLabel = 0;
//     for (const auto& entry : fs::directory_iterator(directory)) {
//         if (entry.is_directory()) {
//             string className = entry.path().filename().string();
//             if (labelMap.find(className) == labelMap.end()) {
//                 labelMap[className] = currentLabel;
//                 reverseLabelMap[currentLabel] = className;
//                 currentLabel++;
//             }
//             int label = labelMap[className];
//             for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
//                 if (imgEntry.path().extension() == ".jpg" || imgEntry.path().extension() == ".png") {
//                     Mat img = imread(imgEntry.path().string(), IMREAD_GRAYSCALE);
//                     if (!img.empty()) {
//                         Mat resizedImg;
//                         resize(img, resizedImg, Size(64, 64));  // Asegurarse de que todas las imágenes sean del mismo tamaño
//                         images.push_back(resizedImg);
//                         labels.push_back(label);
//                     }
//                 }
//             }
//         }
//     }
// }

// int main() {
//     // Directorio de las imágenes de entrenamiento
//     string trainingDir = "dataset-logos";

//     // Cargar imágenes y etiquetas
//     vector<Mat> images;
//     vector<int> labels;
//     map<string, int> labelMap;
//     map<int, string> reverseLabelMap;
//     loadImagesAndLabels(trainingDir, images, labels, labelMap, reverseLabelMap);

//     // Definir los parámetros del HOGDescriptor
//     HOGDescriptor hog(
//         Size(64, 64),   // winSize
//         Size(16, 16),   // blockSize
//         Size(8, 8),     // blockStride
//         Size(8, 8),     // cellSize
//         9               // nbins
//     );

//     // Extraer características usando HOG
//     vector<Mat> descriptors;
//     for (const auto& image : images) {
//         vector<float> descriptor;
//         hog.compute(image, descriptor);
//         descriptors.push_back(Mat(descriptor).clone());
//     }

//     // Convertir descriptores a un solo Mat
//     Mat trainingData;
//     vconcat(descriptors, trainingData);
//     trainingData = trainingData.reshape(1, descriptors.size());  // Reshape para tener una fila por descriptor

//     // Convertir etiquetas a Mat
//     Mat trainingLabels(labels);

//     // Crear y entrenar el SVM
//     Ptr<SVM> svm = SVM::create();
//     svm->setType(SVM::C_SVC);
//     svm->setKernel(SVM::LINEAR);
//     svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//     svm->train(trainingData, ROW_SAMPLE, trainingLabels);

//     // Guardar el modelo entrenado
//     svm->save("svm_logo_model.yml");

//     // Guardar el mapa de etiquetas en un archivo
//     ofstream labelFile("label_map.txt");
//     for (const auto& pair : reverseLabelMap) {
//         labelFile << pair.first << " " << pair.second << endl;
//     }
//     labelFile.close();

//     cout << "Modelo SVM entrenado y guardado en svm_logo_model.yml" << endl;
//     cout << "Mapa de etiquetas guardado en label_map.txt" << endl;
//     cout << "Tamaño del descriptor: " << trainingData.cols << endl;

//     return 0;
// }






#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <map>
#include <fstream>
#include <numeric>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

// Función para cargar imágenes y etiquetas
void loadImagesAndLabels(const string& directory, vector<Mat>& images, vector<int>& labels, map<string, int>& labelMap, map<int, string>& reverseLabelMap) {
    int currentLabel = 0;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_directory()) {
            string className = entry.path().filename().string();
            if (labelMap.find(className) == labelMap.end()) {
                labelMap[className] = currentLabel;
                reverseLabelMap[currentLabel] = className;
                currentLabel++;
            }
            int label = labelMap[className];
            for (const auto& imgEntry : fs::directory_iterator(entry.path())) {
                if (imgEntry.path().extension() == ".jpg" || imgEntry.path().extension() == ".png") {
                    Mat img = imread(imgEntry.path().string(), IMREAD_GRAYSCALE);
                    if (!img.empty()) {
                        Mat resizedImg;
                        resize(img, resizedImg, Size(64, 64));  // Asegurarse de que todas las imágenes sean del mismo tamaño
                        images.push_back(resizedImg);
                        labels.push_back(label);
                    }
                }
            }
        }
    }
}

int main() {
    // Directorio de las imágenes de entrenamiento
    string trainingDir = "dataset-logos";

    // Cargar imágenes y etiquetas
    vector<Mat> images;
    vector<int> labels;
    map<string, int> labelMap;
    map<int, string> reverseLabelMap;
    loadImagesAndLabels(trainingDir, images, labels, labelMap, reverseLabelMap);

    // Definir los parámetros del HOGDescriptor
    HOGDescriptor hog(
        Size(64, 64),   // winSize
        Size(16, 16),   // blockSize
        Size(8, 8),     // blockStride
        Size(8, 8),     // cellSize
        9               // nbins
    );

    // Extraer características usando HOG
    vector<Mat> descriptors;
    for (const auto& image : images) {
        vector<float> descriptor;
        hog.compute(image, descriptor);
        descriptors.push_back(Mat(descriptor).clone());
    }

    // Convertir descriptores a un solo Mat
    Mat trainingData;
    vconcat(descriptors, trainingData);
    trainingData = trainingData.reshape(1, descriptors.size());  // Reshape para tener una fila por descriptor

    // Convertir etiquetas a Mat
    Mat trainingLabels(labels);

    // Crear y entrenar el SVM
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, trainingLabels);

    // Guardar el modelo entrenado
    svm->save("svm_logo_model.yml");

    // Guardar el mapa de etiquetas en un archivo
    ofstream labelFile("label_map.txt");
    for (const auto& pair : reverseLabelMap) {
        labelFile << pair.first << " " << pair.second << endl;
    }
    labelFile.close();

    cout << "Modelo SVM entrenado y guardado en svm_logo_model.yml" << endl;
    cout << "Mapa de etiquetas guardado en label_map.txt" << endl;
    cout << "Tamaño del descriptor: " << trainingData.cols << endl;

    // Evaluación del modelo
    Mat predictedLabels;
    svm->predict(trainingData, predictedLabels);

    // Calcular la matriz de confusión
    int numClasses = reverseLabelMap.size();
    Mat confusionMatrix = Mat::zeros(numClasses, numClasses, CV_32S);
    for (int i = 0; i < trainingLabels.rows; i++) {
        int actual = trainingLabels.at<int>(i);
        int predicted = predictedLabels.at<float>(i);
        confusionMatrix.at<int>(actual, predicted)++;
    }

    // Mostrar la matriz de confusión
    cout << "Matriz de confusión:" << endl;
    cout << confusionMatrix << endl;

    // Calcular y mostrar las métricas
    vector<int> truePositives(numClasses, 0);
    vector<int> falsePositives(numClasses, 0);
    vector<int> falseNegatives(numClasses, 0);

    for (int i = 0; i < numClasses; i++) {
        truePositives[i] = confusionMatrix.at<int>(i, i);
        falsePositives[i] = sum(confusionMatrix.col(i))[0] - truePositives[i];
        falseNegatives[i] = sum(confusionMatrix.row(i))[0] - truePositives[i];
    }

    double accuracy = sum(confusionMatrix.diag())[0] / trainingLabels.rows;
    cout << "Accuracy: " << accuracy << endl;

    for (int i = 0; i < numClasses; i++) {
        double precision = (truePositives[i] + falsePositives[i]) > 0 ? truePositives[i] / double(truePositives[i] + falsePositives[i]) : 0;
        double recall = (truePositives[i] + falseNegatives[i]) > 0 ? truePositives[i] / double(truePositives[i] + falseNegatives[i]) : 0;
        cout << "Clase " << reverseLabelMap[i] << ": Precision = " << precision << ", Recall = " << recall << endl;
    }

    return 0;
}

