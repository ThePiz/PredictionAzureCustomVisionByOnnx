using Microsoft.ML;
using ONNXClassifier.datastructures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ONNXClassifier
{
    class Program
    {
        /// <summary>       
        /// WARN The images are copied to the debug folder at compile time so I put a fixed path        
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            string assetsPath = "assets";//GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "Model", "test.onnx");
           
            var imagesFolder = Path.Combine(assetsPath, "images");

            MLContext mlContext = new MLContext();

            try
            {
                // Load Data
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

                // Create instance of model scorer
                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

                // Use model to score data
                var probabilities = modelScorer.Score(imageDataView);
                Console.WriteLine(Environment.NewLine);

                foreach(var prob in probabilities)
                    Console.WriteLine(prob.ToString());

                Console.WriteLine("Digit any key for exit");
                Console.ReadLine();

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }
    }
}
