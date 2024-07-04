#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <fstream>
#include <map>

using namespace std;
using namespace cv;
using namespace ml;

// Función para cargar el mapa de etiquetas desde un archivo
map<int, string> loadLabelMap(const string& filename) {
    map<int, string> labelMap;
    ifstream file(filename);
    int label;
    string className;
    while (file >> label >> className) {
        labelMap[label] = className;
    }
    return labelMap;
}

int main() {
    // Ruta de la imagen de prueba
    string imagePath = "tiktok.png";

    // Cargar la imagen
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    // Redimensionar la imagen al tamaño esperado
    Mat resizedImage;
    resize(image, resizedImage, Size(64, 64));

    // Cargar el modelo SVM entrenado
    Ptr<SVM> svm = SVM::load("svm_logo_model.yml");

    // Definir los parámetros del HOGDescriptor
    HOGDescriptor hog(
        Size(64, 64),   // winSize
        Size(16, 16),   // blockSize
        Size(8, 8),     // blockStride
        Size(8, 8),     // cellSize
        9               // nbins
    );

    // Extraer características usando HOG
    vector<float> descriptor;
    hog.compute(resizedImage, descriptor);

    // Convertir el descriptor a Mat y asegurar que sea de tipo CV_32F
    Mat descriptorMat = Mat(descriptor).reshape(1, 1);  // Reshape para tener una fila por descriptor
    descriptorMat.convertTo(descriptorMat, CV_32F);     // Convertir a tipo CV_32F

    // Predecir la clase usando el modelo SVM
    int predictedLabel = svm->predict(descriptorMat);

    // Cargar el mapa de etiquetas
    map<int, string> labelMap = loadLabelMap("label_map.txt");

    // Mostrar el nombre de la clase en lugar del número de etiqueta
    if (labelMap.find(predictedLabel) != labelMap.end()) {
        cout << "La imagen ha sido clasificada como: " << labelMap[predictedLabel] << endl;
    } else {
        cout << "Etiqueta desconocida: " << predictedLabel << endl;
    }
    putText(image, labelMap[predictedLabel], Point(50,250), FONT_HERSHEY_SIMPLEX, 1, Scalar (255, 255, 255), 2);
    imshow("Prediccion es: " +labelMap[predictedLabel], image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}


