using OpenCvSharp;
using System;


class Program
{
    static void Main()
    {
        string pathRoot = Path.Combine("..", "..", "..");

        string videoPath = Path.Combine(pathRoot, "Videos","video (2160p).mp4");
        string framesDirectory = Path.Combine(pathRoot, "frames");
        string facialFeaturesFramesDirectory = Path.Combine(pathRoot, "frames_features");
        string classifiersDirectory = Path.Combine(pathRoot, "haarcascades");
        string[] classifiers = new string[]
        {
            "haarcascade_righteye_2splits.xml",
            "haarcascade_lefteye_2splits.xml",
            "haarcascade_nose.xml",
            "haarcascade_mcs_mouth.xml"
        };

        // Paralelizar la detección de rasgos faciales
        Parallel.ForEach(classifiers, classifier =>
        {
            DetectAndSaveFacialFeatures(videoPath, framesDirectory, facialFeaturesFramesDirectory, classifiersDirectory, classifier);
        });

        Console.WriteLine("Proceso de detección de rasgos faciales y generación de imágenes completado.");
    }

    static void DetectAndSaveFacialFeatures(string videoPath, string framesDirectory, string facialFeaturesFramesDirectory, string classifiersDirectory, string classifier)
    {
        string classifierPath = Path.Combine(classifiersDirectory, classifier);
        var cascadeClassifier = new CascadeClassifier(classifierPath);

        using (var videoCapture = new VideoCapture(videoPath))
        {
            if (!videoCapture.IsOpened())
            {
                Console.WriteLine("No se pudo abrir el video.");
                return;
            }

            int framesFor10Seconds = (int)(videoCapture.Fps * 10);
            HashSet<string> savedROIs = new HashSet<string>();

            for (int i = 0; i < framesFor10Seconds; i++)
            {
                using (var frame = new Mat())
                {
                    if (!videoCapture.Read(frame))
                    {
                        Console.WriteLine("No hay más frames en el video. Proceso terminado.");
                        break;
                    }

                    var detections = DetectFeatures(frame, cascadeClassifier, classifier);
                    foreach (var detection in detections)
                    {
                        string roiKey = $"{classifierPath}_{detection.X}_{detection.Y}_{detection.Width}_{detection.Height}";

                        if (savedROIs.Add(roiKey))
                        {
                            Rect roiRect = new Rect(detection.X, detection.Y, detection.Width, detection.Height);
                            Mat roi = new Mat(frame, roiRect);

                            string featureType = Path.GetFileNameWithoutExtension(classifier);
                            string imagePath = Path.Combine(facialFeaturesFramesDirectory, $"{featureType}_{i}.png");
                            Cv2.ImWrite(imagePath, roi);
                        }
                    }

                    string frameImagePath = Path.Combine(framesDirectory, $"facial_features_frame_{i}.png");
                    Cv2.ImWrite(frameImagePath, frame);
                    Console.WriteLine($"Rasgos faciales detectados en frame {i + 1}. Guardado en {frameImagePath}");
                }
            }
        }
    }

    static Rect[] DetectFeatures(Mat frame, CascadeClassifier cascadeClassifier, string classifier)
    {
        // Personalizacion los parámetros de detección según el tipo de rasgo facial
        if (classifier.Equals("haarcascade_righteye_2splits.xml"))
        {
            return cascadeClassifier.DetectMultiScale(frame, 1.2, 60);
        }

        else if (classifier.Equals("haarcascade_lefteye_2splits.xml"))
        {
            return cascadeClassifier.DetectMultiScale(frame, 1.2, 60);
        }

        else if (classifier.Equals("haarcascade_nose.xml"))
        {
            return cascadeClassifier.DetectMultiScale(frame, 1.1, 40);
        }

        else if (classifier.Equals("haarcascade_mcs_mouth.xml"))
        {
            return cascadeClassifier.DetectMultiScale(frame, 1.1, 50);
        }

        return cascadeClassifier.DetectMultiScale(frame, 1.3, 20); // Parámetros predeterminados si no se encuentra una configuración específica
    }
}
